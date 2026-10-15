"""Real HF/PEFT API test with a local randomly initialized microscopic Qwen3.

This is an optional CPU contract test, not pretrained Qwen3-4B training or any
AppWorld evidence. It downloads no model and creates no main-experiment data.
"""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")
tokenizers = pytest.importorskip("tokenizers")

from b2_appworld.learning import create_model_bundle, load_predictor, save_predictor_checkpoint
from b2_appworld.schemas import Candidate, Goal


def test_local_random_qwen3_peft_training_and_adapter_reload(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        base = tmp_path / "random-test-only-qwen3"
        config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                             max_position_embeddings=2048)
        Qwen3Model(config).save_pretrained(base)
        tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "[BOS]": 2, "[EOS]": 3},
                                        unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]",
                                bos_token="[BOS]", eos_token="[EOS]").save_pretrained(base)
        model, tokenizer, metadata = create_model_bundle({
            "backend": "hf", "kind": "effect", "base_model": str(base), "device": "cpu",
            "revision": "random-test-only-local-v1", "quantized": False, "max_length": 2048,
            "gradient_checkpointing": False, "lora_rank": 2, "lora_alpha": 4})
        ids = torch.tensor([[2, 4, 5, 6, 7]])
        before = model.head.weight.detach().clone()
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                       goal_positions=torch.tensor([[2, 4]]))
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 4), torch.tensor([1, 3]))
        loss.backward()
        assert model.head.weight.grad.abs().sum() > 0
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for name, p in model.encoder.named_parameters() if "lora" in name)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        optimizer.step()
        assert not torch.equal(before, model.head.weight.detach())
        checkpoint = tmp_path / "adapter-test-only"
        save_predictor_checkpoint(model, tokenizer, metadata, checkpoint)
        first = load_predictor(checkpoint, device="cpu", quantized=False)
        second = load_predictor(checkpoint, device="cpu", quantized=False)
        goals = (Goal("a", "Set A", "Set A"), Goal("b", "keep B", "keep B"))
        candidates = [Candidate("a", "print(1)", ("a",))]
        a = first.predict("Set A and keep B", [], goals, candidates)
        b = second.predict("Set A and keep B", [], goals, candidates)
        assert a == b and len(a[0]) == 2 and len(a[0][0]) == 4
        assert (checkpoint / "adapter" / "adapter_model.safetensors").is_file()
        assert (checkpoint / "head.pt").is_file()
    finally:
        torch.set_num_threads(old_threads)
