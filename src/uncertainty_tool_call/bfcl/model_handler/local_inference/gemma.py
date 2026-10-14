from bfcl.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl.model_handler.utils import (
    combine_consecutive_user_prompts,
    convert_system_prompt_into_user_prompt,
    func_doc_language_specific_pre_processing,
    system_prompt_pre_processing_chat_model,
)
from overrides import override

import traceback

class GemmaHandler(OSSHandler):
    def __init__(self, model_name, temperature, test_category=None) -> None:
        super().__init__(model_name, temperature, test_category)

    @override
    def _format_prompt(self, messages, function):
        """
        "bos_token": "<bos>",
        "chat_template": "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}",
        """
        formatted_prompt = "<bos>"

        for message in messages:
            formatted_prompt += f"<start_of_turn>{message['role']}\n{message['content'].strip()}<end_of_turn>\n"

        formatted_prompt += f"<start_of_turn>model\n"

        return formatted_prompt

    @override
    def _format_prompt_p_true(self, messages, function, greedy_answer, sampled_answers):
        """
        <bos><start_of_turn>user
        You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
        If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
        You should only return the function calls in your response.

        If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
        You SHOULD NOT include any other text in the response.

        At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.

        Here is a list of functions in JSON format that you can invoke.
        [{'name': 'restaurant_search', 'description': 'Locates top rated restaurants based on specific criteria such as type of cuisine, ratings, and facilities. Note that the provided function is in Python 3 syntax.', 'parameters': {'type': 'dict', 'properties': {'location': {'type': 'string', 'description': 'The city and state, e.g. New York City, NY'}, 'cuisine': {'type': 'string', 'description': 'Preferred type of cuisine e.g., Italian, Indian, American, etc.'}, 'rating': {'type': 'integer', 'description': 'Minimum average customer rating out of 5'}, 'accepts_credit_cards': {'type': 'boolean', 'description': 'If the restaurant should accept credit cards.'}}, 'required': ['location', 'cuisine', 'rating', 'accepts_credit_cards']}}]



        Find me the best Italian restaurants in New York City with average customer ratings of more than 4 and accepts credit cards.<end_of_turn>
        <start_of_turn>model
        """

        formatted_prompt = "<bos>"

        start_of_turn_token = "<start_of_turn>"
        end_of_turn_token = "<end_of_turn>"
        model_turn_name = "model"
        
        true_false_question = "Is the possible answer:\n"
        true_false_question += "A) True\n"
        true_false_question += "B) False\n"
        true_false_question += "Respond with A or B only."

        few_shot_prompt_1 = "Question: What is 19/53?\n"
        few_shot_prompt_1 += "Here is a list of functions in JSON format that can be invoked:\n"
        few_shot_prompt_1 += "[{'name': 'divide', 'description': 'Divides two numbers.', 'parameters': {'type': 'dict', 'properties': {'numerator': {'type': 'float', 'description': 'The numerator of the fraction.'}, 'denominator': {'type': 'float', 'description': 'The denominator of the fraction.'}}, 'required': ['numerator', 'denominator']}}, "
        few_shot_prompt_1 += "{'name': 'add', 'description': 'Adds two integers.', 'parameters': {'type': 'dict', 'properties': {'a': {'type': 'int', 'description': 'The first integer.'}, 'b': {'type': 'int', 'description': 'The second integer.'}}, 'required': ['a', 'b']}}]"
        few_shot_prompt_1 += "\nHere are some brainstormed ideas:\n"
        few_shot_prompt_1 += "[divide(numerator=53, denominator=19)]\n"
        few_shot_prompt_1 += "[divide(denominator=19, numerator=19)]\n"
        few_shot_prompt_1 += "[divide(numerator=53, denominator=53)]\n"
        few_shot_prompt_1 += "[add(numerator=19, denominator=53)]\n"
        few_shot_prompt_1 += f"Possible answer:\n"
        few_shot_prompt_1 += "[divide(numerator=53, denominator=19)]\n"

        few_shot_prompt_2 = "Question: What is 19/53?\n"
        few_shot_prompt_2 += "Here is a list of functions in JSON format that can be invoked:\n"
        few_shot_prompt_2 += "[{'name': 'divide', 'description': 'Divides two numbers.', 'parameters': {'type': 'dict', 'properties': {'numerator': {'type': 'float', 'description': 'The numerator of the fraction.'}, 'denominator': {'type': 'float', 'description': 'The denominator of the fraction.'}}, 'required': ['numerator', 'denominator']}}, "
        few_shot_prompt_2 += "{'name': 'add', 'description': 'Adds two integers.', 'parameters': {'type': 'dict', 'properties': {'a': {'type': 'int', 'description': 'The first integer.'}, 'b': {'type': 'int', 'description': 'The second integer.'}}, 'required': ['a', 'b']}}]"
        few_shot_prompt_2 += "\nHere are some brainstormed ideas:\n"
        few_shot_prompt_2 += "[divide(numerator=19, denominator=53)]\n"
        few_shot_prompt_2 += "[divide(denominator=19, numerator=19)]\n"
        few_shot_prompt_2 += "[divide(numerator=53, denominator=53)]\n"
        few_shot_prompt_2 += "[add(numerator=53, denominator=19)]\n"
        few_shot_prompt_2 += f"Possible answer:\n"
        few_shot_prompt_2 += "[divide(numerator=19, denominator=53)]\n" 


        assert len(messages) == 1
        for message in messages:
    
            # if message['role'] == "system":
            formatted_prompt += f"{start_of_turn_token}{message['role']}\n"
            formatted_prompt += "You are an expert in composing functions. You are given a question and a set of possible functions.\n"
            formatted_prompt += "You are also given brainstormed ideas and a possible answer. " # added
            formatted_prompt += "Based on the question, you have to assess if the possible answer achieves the purpose.\n" # rephrased

            formatted_prompt += "If none of the functions can be used, it should be stated out in the answer. If the given question lacks the parameters required by the function, it should also be pointed out in the answer.\n" # rephrased
            formatted_prompt += "Otherwise, only function calls should be included in the answer.\n" # rephrased
            
            formatted_prompt += "Any invoked function(s) MUST be put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]\n" # rephrased
            
            possible_fct_msg = message['content'].split("Here is a list of functions in JSON format that you can invoke.")[-1]
            user_question = possible_fct_msg.split("\n\n")[-1]
            possible_fct_msg = possible_fct_msg.split("\n\n")[0]
            possible_fct_msg = "Here is a list of functions in JSON format that can be invoked:" + possible_fct_msg # rephrased

            # add this question to follow the "user - model - user" structure
            formatted_prompt += "\nReply with Yes if this is clear."
            formatted_prompt += f"{end_of_turn_token}\n"

            formatted_prompt += f"{start_of_turn_token}{model_turn_name}\n" 
            formatted_prompt += f"Yes{end_of_turn_token}\n"
            
            # elif message['role'] == "user":

            # few-shot question 1
            formatted_prompt += f"{start_of_turn_token}{message['role']}\n"
            formatted_prompt += few_shot_prompt_1
            formatted_prompt += f"{true_false_question}{end_of_turn_token}\n"

            formatted_prompt += f"{start_of_turn_token}{model_turn_name}\n" 
            formatted_prompt += f"The possible answer is:\nB{end_of_turn_token}\n"

            # few-shot question 2
            formatted_prompt += f"{start_of_turn_token}{message['role']}\n"
            formatted_prompt += few_shot_prompt_2
            formatted_prompt += f"{true_false_question}{end_of_turn_token}\n"

            formatted_prompt += f"{start_of_turn_token}{model_turn_name}\n" 
            formatted_prompt += f"The possible answer is:\nA{end_of_turn_token}\n"

            # actual question
            formatted_prompt += f"{start_of_turn_token}{message['role']}\n"
            formatted_prompt += "Question: " + user_question
            formatted_prompt += f"\n{possible_fct_msg}"

            formatted_prompt += "Here are some brainstormed ideas:\n" # taken from p(True) paper
            for a, answer in enumerate([greedy_answer] + list(set(sampled_answers) - {greedy_answer})):
                if a > 0 and len(answer) > 512:
                    continue
                formatted_prompt += f"{answer.strip()}\n"

            formatted_prompt += f"Possible answer:\n"
            formatted_prompt += f"{greedy_answer.strip()}\n"

            formatted_prompt += f"{true_false_question}{end_of_turn_token}\n"

            formatted_prompt += f"{start_of_turn_token}{model_turn_name}\n" 
            formatted_prompt += "The possible answer is:\n"
       
        return formatted_prompt

    @override
    def _format_prompt_abstention(self, messages, greedy_answer):

        formatted_prompt = "<bos>"

        start_of_turn_token = "<start_of_turn>"
        end_of_turn_token = "<end_of_turn>"
        model_turn_name = "model"
        
        true_false_question = "The answer is explicitly abstaining from calling a function:\n"
        true_false_question += "A) Yes\n"
        true_false_question += "B) No\n"
        true_false_question += "Respond with A or B only."


        few_shot_prompts = [
            {
                'question':"What is 19/53?",
                'answer': "```json\n[divide(numerator=19, denominator=53)]\n```",
                'correct': "B" # (correct function call)
            },
            {
                'question':"What is 19/53?",
                'answer': "None of the given functions can be used.",
                'correct': "A"
            },
            {
                'question':"What is 19/53?",
                'answer': "The answer is approximately 0.4",
                'correct': "B" # (direct answer)
            },
            # {
            #     'question':"What is 19/53?",
            #     'answer': "Division is splitting a number into equal parts.",
            #     'correct': "B" # (unrelated answer)
            # },
            # {
            #     'question':"What is 19/53?",
            #     'answer': "I know.",
            #     'correct': "B" # (no abstention)
            # }, 
            # {
            #     'question':"What is 19/53?",
            #     'answer': "I don't know.",
            #     'correct': "A"
            # },
            {
                'question':"What is 19/53?",
                'answer': "[divide(numerator=53, denominator=19)]",
                'correct': "B" # (wrong function call)
            },

        ]

        assert len(messages) == 1
        for message in messages:
           
            possible_fct_msg = message['content'].split("Here is a list of functions in JSON format that you can invoke.")[-1]
            user_question = possible_fct_msg.split("\n\n")[-1]
    
            # if message['role'] == "system":
            formatted_prompt += f"{start_of_turn_token}{message['role']}\n"
            formatted_prompt += "You are an expert in assessing function calls. You are given a question and an answer.\n" # rephrased

            formatted_prompt += "You have to assess if the answer is a explicit abstention from calling a function.\n" # rephrased
            
            formatted_prompt += "An abstention is an answer that deliberately refrains from making a function call due lack of information or knowledge.\n" # added
            
            # add this question to follow the "user - model - user" structure
            formatted_prompt += "\nReply with Yes if this is clear."
            formatted_prompt += f"{end_of_turn_token}\n"

            formatted_prompt += f"{start_of_turn_token}{model_turn_name}\n" 
            formatted_prompt += f"Yes{end_of_turn_token}\n"
            
            # elif message['role'] == "user":

            # few-shot questions
            for few_shot_prompt in few_shot_prompts:

                formatted_prompt += f"{start_of_turn_token}{message['role']}\n"

                formatted_prompt += f"Question: {few_shot_prompt['question'].strip()}\n"
                formatted_prompt += f"Answer:\n"
                formatted_prompt += f"{few_shot_prompt['answer'].strip()}\n"
                formatted_prompt += f"{true_false_question}{end_of_turn_token}\n"

                formatted_prompt += f"{start_of_turn_token}{model_turn_name}\n" 
                formatted_prompt += f"The answer is:\n{few_shot_prompt['correct']}{end_of_turn_token}\n"

            # actual question
            formatted_prompt += f"{start_of_turn_token}{message['role']}\n"
            formatted_prompt += f"Question: {user_question.strip()}\n"
            formatted_prompt += f"Answer:\n"
            formatted_prompt += f"{greedy_answer.strip()}\n"

            formatted_prompt += f"{true_false_question}{end_of_turn_token}\n"

            formatted_prompt += f"{start_of_turn_token}{model_turn_name}\n" 
            formatted_prompt += "The answer is:\n"
       
        return formatted_prompt


    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_category
        )

        for round_idx in range(len(test_entry["question"])):
            test_entry["question"][round_idx] = convert_system_prompt_into_user_prompt(
                test_entry["question"][round_idx]
            )
            test_entry["question"][round_idx] = combine_consecutive_user_prompts(
                test_entry["question"][round_idx]
            )
            test_entry["question"][round_idx] = self._substitute_prompt_role(
                test_entry["question"][round_idx]
            )

        return {"message": [], "function": functions}

    @staticmethod
    def _substitute_prompt_role(prompts: list[dict]) -> list[dict]:
        for prompt in prompts:
            if prompt["role"] == "assistant":
                prompt["role"] = "model"

        return prompts
