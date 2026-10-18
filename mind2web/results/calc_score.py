import json
import os
from functools import reduce
import argparse


def main(results_dir: str):
    for bench in ["test_domain", "test_task", "test_website"]:
        path = os.path.join(results_dir, bench)
        if not os.path.exists(path):
            continue

        files = os.listdir(path)
        file_paths = sum([
            [*map(
                lambda x: os.path.join(path, f, "workflow", x),
                os.listdir(os.path.join(path, f, "workflow")))
            ]
            for f in files
        ], [])

        jss = [json.load(open(file_path, "r")) for file_path in file_paths]

        for js in jss:
            l = len(js)

            for ii, (x, y) in enumerate(zip(range(0, l, 2), range(1, l, 2))):
                try:
                    pred = js[x]["output"].lower().split()
                    label = js[y]["target_act"].lower().split()

                    if "action:" in pred:
                        pred = pred[pred.index("action:")+1:]

                        action = pred[0] if "`" not in pred[0] else pred[0][1:]
                        element = pred[1] if "`" not in pred[1] else pred[1][:-1]

                        a_acc = int(action in label[0])
                        e_acc = int(element in label[1])

                        if len(pred) > 2 and len(label) > 2 and 'type' in label[0] and 'type' in pred[0]:
                            step_sr = 0
                        else:
                            step_sr = int(a_acc and e_acc)

                    else:
                        a_acc = 0
                        e_acc = 0
                        step_sr = 0

                    if not js[-1]["element_acc"][ii]:
                        js[-1]["element_acc"][ii] = a_acc
                    if not js[-1]["action_f1"][ii]:
                        js[-1]["action_f1"][ii] = e_acc
                    if not js[-1]["step_success"][ii]:
                        js[-1]["step_success"][ii] = step_sr
                except:
                    continue

            if not js[-1]["success"][0]:
                js[-1]["success"][0] = reduce(lambda x, y: x & y, js[-1]["step_success"])

        f = lambda x, y: print(x, round(y*100, 1))

        print(bench)

        e_accs = sum([*map(lambda x: x[-1]["element_acc"], jss)], [])
        f("Element Acc:", sum(e_accs) / len(e_accs))

        a_f1 = sum([*map(lambda x: x[-1]["action_f1"], jss)], [])
        f("Action F1:  ", sum(a_f1) / len(a_f1))

        stepsr = sum([*map(lambda x: x[-1]["step_success"], jss)], [])
        f("Step SR:    ", sum(stepsr) / len(stepsr))

        sr = [*map(lambda x: x[-1]["success"][0], jss)]
        f("SR:         ", sum(sr) / len(sr))

        print()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    args = parser.parse_args()

    main(args.results_dir)