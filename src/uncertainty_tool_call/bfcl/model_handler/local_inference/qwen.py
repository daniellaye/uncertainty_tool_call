from bfcl.model_handler.local_inference.base_oss_handler import OSSHandler
from overrides import override

import traceback

class QwenHandler(OSSHandler):
    def __init__(self, model_name, temperature, test_category=None) -> None:
        super().__init__(model_name, temperature, test_category)

    @override
    def _format_prompt(self, messages, function):
        # Qwen is using its prompting mode, not the tool use mode
        """
        "chat_template": "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n",
        """
        formatted_prompt = ""

        for message in messages:
            formatted_prompt += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"

        formatted_prompt += "<|im_start|>assistant\n"

        return formatted_prompt

    @override
    def _format_prompt_p_true(self, messages, function, greedy_answer, sampled_answers):
        """
        ### Original system prompt ###
        You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more answers to achieve the purpose.                                                
        If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.                                                                                                                    
        You should only return the function calls in your response.                                                                                                                                                                                                    
                                                                                                                                                                                                                                                                    
        If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]                                                                                   
        You SHOULD NOT include any other text in the response.                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                    
        At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.                                                                                                                                                                   
                                                                                                                                                                                                                                                                    
        Here is a list of functions in JSON format that you can invoke.  
        ...
        """

        formatted_prompt = ""

        start_of_turn_token = "<|im_start|>"
        end_of_turn_token = "<|im_end|>"
        model_turn_name = "assistant"
        
        true_false_question = "Is the possible answer:\n"
        true_false_question += "A) True\n"
        true_false_question += "B) False\n"
        true_false_question += "Respond with A or B only."

        few_shot_prompt_1 = "Question: What is 19/53?\n\n"
        few_shot_prompt_1 += "Here is a list of functions in JSON format that can be invoked:\n"
        few_shot_prompt_1 += "[{'name': 'divide', 'description': 'Divides two numbers.', 'parameters': {'type': 'dict', 'properties': {'numerator': {'type': 'float', 'description': 'The numerator of the fraction.'}, 'denominator': {'type': 'float', 'description': 'The denominator of the fraction.'}}, 'required': ['numerator', 'denominator']}}, "
        few_shot_prompt_1 += "{'name': 'add', 'description': 'Adds two integers.', 'parameters': {'type': 'dict', 'properties': {'a': {'type': 'int', 'description': 'The first integer.'}, 'b': {'type': 'int', 'description': 'The second integer.'}}, 'required': ['a', 'b']}}]"
        few_shot_prompt_1 += "\n\nHere are some brainstormed ideas:\n\n"
        few_shot_prompt_1 += "[divide(numerator=53, denominator=19)]\n\n"
        few_shot_prompt_1 += "[divide(denominator=19, numerator=19)]\n\n"
        few_shot_prompt_1 += "[divide(numerator=53, denominator=53)]\n\n"
        few_shot_prompt_1 += "[add(numerator=19, denominator=53)]\n\n"
        few_shot_prompt_1 += f"Possible answer:\n\n"
        few_shot_prompt_1 += "[divide(numerator=53, denominator=19)]\n\n"

        few_shot_prompt_2 = "Question: What is 19/53?\n\n"
        few_shot_prompt_2 += "Here is a list of functions in JSON format that can be invoked:\n"
        few_shot_prompt_2 += "[{'name': 'divide', 'description': 'Divides two numbers.', 'parameters': {'type': 'dict', 'properties': {'numerator': {'type': 'float', 'description': 'The numerator of the fraction.'}, 'denominator': {'type': 'float', 'description': 'The denominator of the fraction.'}}, 'required': ['numerator', 'denominator']}}, "
        few_shot_prompt_2 += "{'name': 'add', 'description': 'Adds two integers.', 'parameters': {'type': 'dict', 'properties': {'a': {'type': 'int', 'description': 'The first integer.'}, 'b': {'type': 'int', 'description': 'The second integer.'}}, 'required': ['a', 'b']}}]"
        few_shot_prompt_2 += "\n\nHere are some brainstormed ideas:\n\n"
        few_shot_prompt_2 += "[divide(numerator=19, denominator=53)]\n\n"
        few_shot_prompt_2 += "[divide(denominator=19, numerator=19)]\n\n"
        few_shot_prompt_2 += "[divide(numerator=53, denominator=53)]\n\n"
        few_shot_prompt_2 += "[add(numerator=53, denominator=19)]\n\n"
        few_shot_prompt_2 += f"Possible answer:\n\n"
        few_shot_prompt_2 += "[divide(numerator=19, denominator=53)]\n\n" 

        
        for message in messages:
            if message['role'] == "system":
                msg = "You are an expert in composing functions. You are given a question and a set of possible functions.\n"
                msg += "You are also given brainstormed ideas and a possible answer. " # added
                msg += "Based on the question, you have to assess if the possible answer achieves the purpose.\n\n" # rephrased

                msg += "If none of the functions can be used, it should be stated out in the answer. If the given question lacks the parameters required by the function, it should also be pointed out in the answer.\n" # rephrased
                msg += "Otherwise, only function calls should be included in the answer.\n\n" # rephrased
                
                msg += "Any invoked function(s) MUST be put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]\n" # rephrased
                
                possible_fct_msg = "Here is a list of functions in JSON format that can be invoked:" # rephrased
                possible_fct_msg += message['content'].split("Here is a list of functions in JSON format that you can invoke.")[-1]

                formatted_prompt += f"{start_of_turn_token}{message['role']}\n{msg}{end_of_turn_token}\n"
            
            elif message['role'] == "user":

                # few-shot question 1
                formatted_prompt += f"{start_of_turn_token}{message['role']}\n"
                formatted_prompt += few_shot_prompt_1
                formatted_prompt += f"{true_false_question}{end_of_turn_token}\n"

                formatted_prompt += f"{start_of_turn_token}{model_turn_name}\n" 
                formatted_prompt += f"The possible answer is: B{end_of_turn_token}\n"

                # few-shot question 2
                formatted_prompt += f"{start_of_turn_token}{message['role']}\n"
                formatted_prompt += few_shot_prompt_2
                formatted_prompt += f"{true_false_question}{end_of_turn_token}\n"

                formatted_prompt += f"{start_of_turn_token}{model_turn_name}\n" 
                formatted_prompt += f"The possible answer is: A{end_of_turn_token}\n"

                # actual question
                formatted_prompt += f"{start_of_turn_token}{message['role']}\n"
                formatted_prompt += "Question: " + message['content']
                formatted_prompt += f"\n\n{possible_fct_msg}"

                formatted_prompt += "Here are some brainstormed ideas:\n\n" # taken from p(True) paper
                for a, answer in enumerate([greedy_answer] + list(set(sampled_answers) - {greedy_answer})):
                    if a > 0 and len(answer) > 512:
                        continue
                    formatted_prompt += f"{answer.strip()}\n\n"

                formatted_prompt += f"Possible answer:\n\n"
                formatted_prompt += f"{greedy_answer.strip()}\n\n"

                formatted_prompt += f"{true_false_question}{end_of_turn_token}\n"

                formatted_prompt += f"{start_of_turn_token}{model_turn_name}\n" 
                formatted_prompt += "The possible answer is: "
        
        return formatted_prompt