import transformers
import google.generativeai as genai
import google
from tqdm.auto import tqdm

import json, argparse, random, time

from utils import generate_few_shot, wcst_generator, string_generator

wcst_prompt = """You are performing the Wisconsin Card Sorting Test (WCST).
You will be shown a given card with a symbol on it, and you will have to match it to one of four option cards according to an attribute that you have to figure out.
The cards will be described by the following attributes:
1. Number of symbols
2. Color of symbols
3. Shape of symbols

You will be told "Correct!" if you are correct and "Incorrect. Please try again." if you are incorrect.
If you are incorrect, you either made a mistake or the rule has changed. You have to figure out the correct rule to match the cards.
If you are correct, you have to stick with the same attribute until you are incorrect.
There is always a true answer in the task, and you have to keep performing the task until the end of the test.
Your final answer should be a number between 1-4 corresponding to the index of the card you think is the correct match.
State your final answer using the template: "ANSWER: <index>"

"""
wcst_random_prompt = """You are performing the Wisconsin Card Sorting Test (WCST).
You will be shown a given card with a symbol on it, and you will have to match it to one of four option cards according to an attribute that you have to figure out.
The cards will be described by the following attributes in a random order:
1. Number of symbols
2. Color of symbols
3. Shape of symbols

You will be told "Correct!" if you are correct and "Incorrect. Please try again." if you are incorrect.
If you are incorrect, you either made a mistake or the rule has changed. You have to figure out the correct rule to match the cards.
If you are correct, you have to stick with the same attribute until you are incorrect.
There is always a true answer in the task, and you have to keep performing the task until the end of the test.
Your final answer should be a number between 1-4 corresponding to the index of the card you think is the correct match.
State your final answer using the template: "ANSWER: <index>"

"""
random_prompt = """You are performing a modified version of the Wisconsin Card Sorting Test (WCST).
You will be shown a given string, and you have to match it with one of four option strings according to a rule that you have to figure out.
The rule is one of the following:
1. Length of the string
2. The number of vowels in the string
3. The number of consonants in the string

You will be told "Correct!" if you are correct and "Incorrect. Please try again." if you are incorrect.
If you are incorrect, you either made a mistake or the rule has changed. You have to figure out the correct rule to match the strings.
If you are correct, you have to stick with the same rule until you are incorrect.
There is always a true answer in the task, and you have to keep performing the task until the end of the test.
Your final answer should be a number between 1-4 corresponding to the index of the string you think is the correct match.
State your final answer using the template: "ANSWER: <index>"

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama")
    parser.add_argument("--variant", type=str, default="card")
    parser.add_argument("--max_trials", type=int, default=64)
    parser.add_argument("--num_correct", type=int, default=5)
    parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--verbose", type=int, default=15)
    
    args = parser.parse_args()
    variant = args.variant
    max_trials = args.max_trials
    num_correct = args.num_correct
    few_shot = args.few_shot
    cot = args.cot

    if variant == "card":
        system_prompt = wcst_prompt
        rules = ["color", "shape", "number"]
    elif variant == "card-random":
        system_prompt = wcst_random_prompt
        rules = ["color", "shape", "number"]
    elif variant == "string":
        system_prompt = random_prompt
        rules = ["length", "vowels", "consonants"]
    else:
        raise Exception("Variant not recognized")

    if few_shot:
        system_prompt += generate_few_shot(variant)

    if cot:
        system_prompt += "Explain your thought process regarding the problem and the feedbacks you received.\n"

    if args.model == "llama":
        pipeline = transformers.pipeline(
        "text-generation",
        model="meta-llama/Llama-3.1-8B-Instruct",
        )

        messages = [{"role": "system", "content": system_prompt},]

    elif args.model == "gemini":
        # api_key = os.getenv("GOOGLE_API_KEY")
        # genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        chat = model.start_chat(
            history=[
                {"role": "user", "parts": "I will explain how you should behave.\n" + system_prompt},
                {"role": "model", "parts": "Understood. Let's start the test.\n"},
                ]
        )
    else:
        raise Exception("Model not recognized")
    
    save_path = f"{args.model}_{variant}_{max_trials}-{num_correct}.json"
    if few_shot:
        save_path = save_path.replace(".json", "_few_shot.json")
    if cot:
        save_path = save_path.replace(".json", "_cot.json")

    save = []
    
    n_trials = 0
    completed_cat = 0
    total_correct = 0
    correct_prefix = ""

    with tqdm(total=max_trials, desc="Total trials") as pbar:
        for _ in range(2):      
            for rule in rules:
                correct_cnt = 0
                
                for trial in tqdm(range(num_correct), total=num_correct, desc=f"Correct responses for {rule}"):
                    if variant == "card":
                        given, opt = wcst_generator(rule, False)
                    elif variant == "card-random":
                        given, opt = wcst_generator(rule, True)
                    else:
                        given, opt = string_generator(rule)

                    chosen = opt[0]
                    random.shuffle(opt)
                    chosen_idx = opt.index(chosen) + 1
    
                    test_prompt = f"""Given: {given}
Options:
1. {opt[0]}
2. {opt[1]}
3. {opt[2]}
4. {opt[3]}
"""
    
                    correct = False
                    while not correct:
                        if n_trials >= max_trials:
                            break
                        pbar.update(1)
    
                        n_trials += 1

                        if args.model == "llama":
                            messages.append({"role": "user", "content": correct_prefix + test_prompt})

                            messages = pipeline(messages, 
                                                max_new_tokens=500,
                                                pad_token_id=pipeline.tokenizer.eos_token_id)[0]["generated_text"]
                            response = messages[-1]["content"]
                            ans = response.split("ANSWER: ")[-1].strip()

                        elif args.model == "gemini":
                            try:
                                response = chat.send_message(correct_prefix + test_prompt).text
                            except google.api_core.exceptions.ResourceExhausted:
                                time.sleep(60)
                                response = chat.send_message(correct_prefix + test_prompt).text
                            ans = response.split("ANSWER: ")[-1].strip()
                            time.sleep(2)

                        if len(ans) != 1:
                            correct_prefix = """Answer not found. Please state your answer using the template: \"ANSWER: <index>\""""
                        elif ans == str(chosen_idx):
                            correct_prefix = "Correct!\n"
                            correct = True
                            correct_cnt += 1
                            total_correct += 1
                        else:
                            correct_prefix = "Incorrect. Please try again.\n"

                        if n_trials % 15 == 0:
                            tqdm.write(test_prompt)
                            tqdm.write(response)

                        save_row = {"rule": rule,
                                    "correct": correct,
                                    "question": test_prompt,
                                    "response": response,
                                    "model_ans": ans,
                                    "true_ans": chosen_idx}
                        save.append(save_row)
                        with open(save_path, "w") as f:
                            json.dump(save, f, indent=4)
                
                if correct_cnt == num_correct:
                    completed_cat += 1

    print(f"Completed categories: {completed_cat}")
    print(f"Total number of trials: {n_trials}")
    print(f"Total accuracy: {total_correct/n_trials}")