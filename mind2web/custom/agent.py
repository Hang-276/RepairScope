import argparse
import json
import re
from typing import Any, Optional
import numpy as np

from beartype import beartype

from agent.prompts import *
from browser_env import Trajectory
from browser_env.actions import (
    Action,
    ActionParsingError,
    create_id_based_action,
    create_none_action,
    create_playwright_action,
)
from browser_env.utils import Observation, StateInfo
from llms import (
    call_llm,
    generate_from_huggingface_completion,
    generate_from_openai_completion,
    lm_config,
)
# generate_from_openai_chat_completion
from browser_env.env_config import URL_MAPPINGS
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import OpenAI, ChatOpenAI
from llms.tokenizers import Tokenizer
from pprint import pprint

import sys
import os
from pathlib import Path

from agent.agent import Agent
from agent.prompts.prompt_constructor import PromptConstructor, CoTPromptConstructor
#import weave

#weave.init("world_model")

class AWMWMAgent(Agent):
    """
    World Model Agent Process:
    1. Sample Multiple Actions:
       Generate a set of potential actions.

    2. Predict Next States:
       For each generated action, use the world model to predict the resulting state.

    3. Calculate Rewards:
       Evaluate the reward for each action based on its predicted next state.

    4. Select Best Action:
       Choose the action with the highest calculated reward.
    """
    @beartype
    def __init__(
        self,
        action_prediction_prompt_path: str,
        state_prediction_prompt_path: str,
        value_function_prompt_path: str,
        model_name: str,
        branching_factor: int,
        action_set_tag: str,
        vf_budget: int,
        world_model_training: bool,
        world_model_name: str | None = None,
        world_model_url: str | None = None,
        value_model_training: bool = False,
        value_model_name: str | None = None,
        value_model_url: str | None = None,
        # tokenizer: Tokenizer,
        # is_multimodal: bool,
        # agent_observation_type: str
    ) -> None:
        super().__init__()
        self.prompt_constructor = CoTPromptConstructor
        self.model_name = model_name
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.action_predction_prompt_path = action_prediction_prompt_path
        self.action_prediction_template = json.load(open(action_prediction_prompt_path))
        self.state_prediction_prompt_path = state_prediction_prompt_path
        self.state_prediction_template = json.load(open(state_prediction_prompt_path))
        self.raw_response_stack = []
        self.intent_stack = []
        self.branching_factor = branching_factor
        self.value_function_prompt_path = value_function_prompt_path
        self.vf_budget = vf_budget
        self.world_model_training = world_model_training
        self.world_model_name = world_model_name
        self.world_model_url = world_model_url
        self.value_model_training = value_model_training
        self.value_model_name = value_model_name
        self.value_model_url = value_model_url
        ##################
        # TODO: remove hardcoded decoding parameters. e.g., top_p and temperature, model_name, base_url.
        ##################
        self.action_set_tag = action_set_tag

        # TODO: add argument for model_name & base_url
        if self.world_model_training:
            self.prediction_llm = ChatOpenAI(
                api_key=self.api_key,
                model_name = self.world_model_name,
                top_p=1.0,
                temperature=1.,
                base_url=self.world_model_url
            )
        else:
            self.prediction_llm = ChatOpenAI(
                api_key=self.api_key,
                model_name=self.model_name,
                top_p=1.0,
                temperature=1.
            )

        # print(self.value_model_training.__class__, self.value_model_training)
        if self.value_model_training:
            self.value_function_llm = ChatOpenAI(
                api_key=self.api_key,
                model_name=self.value_model_name,
                top_p=1.0,
                temperature=1.,
                base_url=self.value_model_url,
                n=self.vf_budget
            )
        else:
            self.value_function_llm = ChatOpenAI(
                api_key=self.api_key,
                model_name=self.model_name,
                n=self.vf_budget
            )
            # self.vlm_client = GPT4V_Client(api_key=self.api_key, model_name=self.model_name)

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    def get_current_observation(self, trajectory: Trajectory) -> str:
        return trajectory[-1]["observation"]['text']

    @beartype
    def next_action(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any],
        actions: list[str],
        branching_factor: int = 5
    ):

        # Step 1: Sample Mulitple actions.
        # Step 2: For each generated action, we predict the next state with the world model.
        # Step 3: We calculate the reward for each action based on the predicted next state.
        # Step 4: We select the action with the highest reward.

        # ==============================
        # Step 1 : Sample multiple actions.
        # ==============================
        # 1-1: load prompt for generating multiple actions
        # 1-2: generate action by calling the openai client instance
        # 1-3: parse the set of actions

        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]
        obs = state_info["observation"]
        page = state_info["info"]["page"]
        raw_url = page.url
        current_url = self.map_url_to_real(url=raw_url)
        # previous_state_prediction = records["steps"][-2]["state_prediction"] if step_idx != 0 else records["steps"][-1]["state_prediction"]

        from mind2web.utils.llm import extract_from_response

        all_actions = {}
        parsed_actions_count = {}
        for response in actions:
            parsed_response = extract_from_response(response, "`")
            if parsed_response in all_actions: # when we get the same action, we increment the count.
                parsed_actions_count[parsed_response] += 1

            else: # when we get a new action, we create a new action instance.
                try:
                    if self.action_set_tag == "id_accessibility_tree":
                        action = create_id_based_action(parsed_response)
                    elif self.action_set_tag == "playwright":
                        action = create_playwright_action(parsed_response)
                    elif self.action_set_tag == "som":
                        action = create_id_based_action(parsed_response)
                    else:
                        raise ValueError(
                            f"Unknown action type {self.action_set_tag}"
                        )
                except Exception:
                    action = create_none_action()

                parsed_actions_count[parsed_response] = 1
                action["raw_prediction"] = response
                all_actions[parsed_response] = action


        top_actions = sorted(parsed_actions_count, key=parsed_actions_count.get, reverse=True)[:branching_factor]
        top_action_count = sum([parsed_actions_count[action] for action in top_actions])
        updated_actions = []
        for action in top_actions:
            a = all_actions[action]
            a['prob'] = parsed_actions_count[action] / top_action_count
            updated_actions.append(a)
        # This part is commented out since the best action index is hardcoded as index=0
        # action = create_id_based_action(parsed_actions[0])
        # action["raw_prediction"] = raw_response_for_action_prediction[0]

        # ==============================
        # Step 2: For each generated action, we predict the next state with the world model.
        # ==============================
        # 2-1: load prompt for the world model
        # 2-2: generate the prediction of the next state with the world model (in qa format)
        # 2-3: parse the qa pairs

        ################
        # predict the next state with the world model.
        ################
        # print("#################### PREDICT NEXT STATE ########################\n")
        state_prediction_prompt = self.generate_prompt(self.state_prediction_template)
        state_prediction_chain = state_prediction_prompt | self.prediction_llm

        if (len(meta_data["action_history"]) != 0):
            previous_action_str = "\n".join(meta_data["action_history"])
        elif (len(meta_data["action_history"]) == 0):
            previous_action_str = "None"

        multiple_input_for_state = []
        for action_ in top_actions:
            input_variable = {
                "objective": intent,
                "url": current_url,
                "previous_action": previous_action_str,
                "observation": obs['text'],
                # "previous_state_prediction": previous_state_prediction,
                "current_action": action_
            }
            multiple_input_for_state.append(input_variable)

        raw_response_for_state_prediction = state_prediction_chain.batch(multiple_input_for_state)
        parsed_state = []
        for response in raw_response_for_state_prediction:
            rationale, next_state = self.extract_state(response.content)
            parsed_state.append(next_state)

        # ==============================
        # Step 3: We calculate the reward for each action based on the predicted next state.
        # ==============================
        # 3-1: load prompt for the value function (text-only)
        # 3-2: inference the reward value with the value function
        # 3-3: parse the value scores.

        value_scores, raw_response_for_value_score_calculation = self.value_function(top_actions, parsed_state, previous_action_str, intent, trajectory)
        # ==============================
        # Step 4: We select the action with the highest reward.
        # ==============================

        best_action_index = np.argmax(value_scores)
        # show scores for each action and parsed state
        # ============================== DEBUGGING =============================
        # for action_index, action in enumerate(top_actions):
        #     print(f"Action: {action}, Score: {value_scores[action_index]}, Next State: {parsed_state[action_index]}")
        # ============================== DEBUGGING =============================
        try:
            action = create_id_based_action(top_actions[best_action_index])
        except:
            action = create_none_action()
        action["raw_prediction"] = actions[best_action_index]

        return (
            action,
            top_actions,
            [ns.content for ns in raw_response_for_state_prediction],
            value_scores,
            raw_response_for_value_score_calculation
        )


    def value_function(
        self,
        predicted_actions: list[str],
        predicted_next_states: list[str],
        previous_action_str: str,
        objective: str,
        trajectory: Trajectory
    ) -> list[float]:
        with open(self.value_function_prompt_path, "r") as f:
            prompt_template = json.load(f)

        value_function_prompt = self.generate_prompt(prompt_template)



        # value_function_chain = value_function_prompt | self.value_function_llm
        multiple_input_for_value_calculation = []
        # print(len(predicted_next_states))

        all_value_scores = []
        all_raw_responses = []
        for action_index, ns in enumerate(predicted_next_states):
            url = trajectory[-1]['url']
            input_variables = {
                "url": url,
                "objective": objective,
                "previous_action": previous_action_str,
                "current_action": predicted_actions[action_index],
                "observation": self.get_current_observation(trajectory),
                "next_state_prediction": ns

            }
            value_function_input = value_function_prompt.invoke(input_variables)

            # multiple_input_for_value_calculation.append(input_variables)

            raw_response_for_value_calculation = self.value_function_llm.generate([value_function_input])
            all_responses = []
            for response in raw_response_for_value_calculation.generations[0]:
                all_responses.append(response.text)

            if self.value_model_training:
                calculated_value_scores, all_individual_value_scores = self.process_mean_value_score_for_value_model(all_responses)
            else:
                calculated_value_scores, all_individual_value_scores = self.process_mean_value_score_likert(all_responses)
            all_value_scores.append(calculated_value_scores)
            all_raw_responses.append(
                [f"Score: {individual_value_score} | {raw_response}" for raw_response, individual_value_score in zip(all_responses, all_individual_value_scores)]
            )

        return all_value_scores, all_raw_responses


    def generate_prompt(self, prompt_template: dict[str, Any]) -> ChatPromptTemplate:
        system_message = prompt_template['intro']
        examples = prompt_template['examples']
        template = prompt_template['template']

        messages = [("system", system_message)]
        if prompt_template != self.state_prediction_template or not self.world_model_training:
            for i in range(len(examples)):
                messages.extend([
                    ("user", examples[i][0]),
                    ("assistant", examples[i][1]),
                ])
        messages.append(("user", template))
        final_prompt = ChatPromptTemplate.from_messages(messages)
        return final_prompt

    def reset(self, test_config_file: str) -> None:
        pass

    def flush_stacks(self):
        self.state_image_stack = []
        self.raw_response_stack = []
        self.intent_stack = []

    def map_url_to_real(self, url: str) -> str:
        """Map the urls to their real world counterparts"""
        for i, j in URL_MAPPINGS.items():
            if i in url:
                url = url.replace(i, j)
        return url

    def extract_action(self, response: str) -> str:
        action_splitter = self.action_prediction_template["meta_data"]["action_splitter"]
        pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"

        action_splitter_2 = "`"
        pattern2 = rf"{action_splitter_2}((.|\n)*?){action_splitter_2}"

        match = re.search(pattern, response)
        match_2 = re.search(pattern2, response)
        if match:
            return match.group(1).strip()
        elif match_2:
            return match_2.group(1).strip()
        else:

            print(f"Cannot parse action from response {response}")
            return "None"
            # raise ActionParsingError(
            #     f"Cannot parse action from response {response}"
            # )

    def extract_state(self, response: str) -> str:
        rationale_pattern = r"\[Rationale\](.*?)\[Next State\]"
        next_state_pattern = r"\[Next State\](.*)"

        rationale_match = re.search(rationale_pattern, response, re.DOTALL)
        next_state_match = re.search(next_state_pattern, response, re.DOTALL)

        rationale = rationale_match.group(1).strip() if rationale_match else ""
        next_state = next_state_match.group(1).strip() if next_state_match else ""

        return rationale, next_state

    def process_mean_value_score(
        self,
        all_responses: list[str],
        should_log: bool = False
    ) -> float:
        """
        Description:
        This method calculates the mean value score from multiple natural language outputs.
        Basically, the LLM is asked to evaluate:
        (1) whether the task is successfully completed at the next state, (i.e. the ouptut contains 'success')
        (2) whether the progress is goring to the correct direction. (i.e. the LLM answers yes to the second question)

        Format:
        input:
        - all_responses (list of strings): list of natural language outputs from the LLM.
        - should_log (boolean): whether to print the intermediate outputs.
        output:
        - score (float): mean value score which is processed from the nl outputs from the LLM.
        """
        all_scores = []
        for r_idx, r in enumerate(all_responses):
            if should_log:
                print(f"Output {r_idx}: {r}")
            try:
                pred = re.search(r'Status: "?(.+)"?', r).group(1)
                if 'success' in pred.lower():
                    score = 1.0
                else:
                    # Check if it's on the path to success
                    on_path = re.search(r'On the right track to success: "?(.+)"?', r).group(1)
                    if 'yes' in on_path.lower():
                        score = 0.5
                    else:
                        score = 0.0
            except Exception as e:
                print(f"Error parsing response: {e}")
                score = 0.0

            all_scores.append(score)

        score = np.mean(all_scores).item()
        if should_log:
            print(f"Final score: {score}")
            print('=' * 30)

        return score, all_scores

    def process_mean_value_score_likert(
        self,
        all_responses: list[str],
        should_log: bool = False
    ) -> float:
        """
        Description:
        This method calculates the mean value score from multiple natural language outputs.
        The LLM is asked to evaluate:
        (1) the performance of an action, using the Licurt Scale (1 to 5),
        (2) whether the action moves the task towards completion (success or failure).

        The method extracts the score from the Licurt Scale for each response and computes the average.

        Format:
        input:
        - all_responses (list of strings): list of natural language outputs from the LLM.
        - should_log (boolean): whether to print the intermediate outputs.
        output:
        - score (float): mean value score which is processed from the nl outputs from the LLM.
        """
        all_scores = []
        for r_idx, r in enumerate(all_responses):
            if should_log:
                print(f"Output {r_idx}: {r}")
            try:
                # Extract the Licurt Scale score from the response
                score = re.search(r'\[Score\]\:\s*(\d+)', r).group(1)
                score = float(score)

                # Ensure the score is within the valid range (1 to 5)
                if 1.0 <= score <= 5.0:
                    all_scores.append(score)
                else:
                    print(f"Warning: Score out of range in response {r_idx}: {score}")
                    all_scores.append(0.0)  # Invalid score

            except Exception as e:
                print(f"Error parsing response {r_idx}: {e}")
                all_scores.append(0.0)  # Default to 0 if there's a parsing issue

        # Calculate the mean of the scores
        mean_score = sum(all_scores) / len(all_scores) if all_scores else 0.0



        if should_log:
            print(f"Final score: {mean_score}")
            print('=' * 30)

        return mean_score, all_scores

    def process_mean_value_score_for_value_model(
        self,
        all_responses: list[str],
        should_log: bool = False
    ) -> float:
        """
        Description:
        This method calculates the mean value score from multiple natural language outputs.
        The LLM is asked to evaluate:
        (1) the performance of an action, using the Licurt Scale (1 to 5),
        (2) whether the action moves the task towards completion (success or failure).

        The method extracts the score from the Licurt Scale for each response and computes the average.

        Format:
        input:
        - all_responses (list of strings): list of natural language outputs from the LLM.
        - should_log (boolean): whether to print the intermediate outputs.
        output:
        - score (float): mean value score which is processed from the nl outputs from the LLM.
        """
        all_scores = []
        for r_idx, r in enumerate(all_responses):
            if should_log:
                print(f"Output {r_idx}: {r}")
            try:
                # Extract the Licurt Scale score from the response
                score = re.search(r'(?:\[Score\]\s*([\d\.]+)|Score\:\s*([\d\.]+))', r).group(1)
                score = float(score)


                # Ensure the score is within the valid range (1 to 5)
                if 0.0 <= score <= 1.0:
                    all_scores.append(score)
                else:
                    print(f"Warning: Score out of range in response {r_idx}: {score}")
                    all_scores.append(0.0)  # Invalid score

            except Exception as e:
                print(f"Error parsing response {r_idx}: {e}")
                all_scores.append(0.0)  # Default to 0 if there's a parsing issue

        # Calculate the mean of the scores
        mean_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

        if should_log:
            print(f"Final score: {mean_score}")
            print('=' * 30)

        return mean_score, all_scores