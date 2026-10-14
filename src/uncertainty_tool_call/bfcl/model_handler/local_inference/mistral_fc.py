import json
import random
import string

from uncertainty_tool_call.bfcl.model_handler.local_inference.base_oss_handler import OSSHandler
from uncertainty_tool_call.bfcl.model_handler.utils import (
    convert_to_function_call,
    func_doc_language_specific_pre_processing,
)
from overrides import override

import traceback

class MistralFCHandler(OSSHandler):
    def __init__(self, model_name, temperature, test_category=None) -> None:
        super().__init__(model_name, temperature, test_category)

    @override
    def decode_ast(self, result, language="Python"):
        # The input is already a list of dictionaries, so no need to decode
        # `[{func1:{param1:val1,...}},{func2:{param2:val2,...}}]`
        if type(result) != list:
            return []
        return result

    @override
    def decode_execute(self, result):
        if type(result) != list:
            return []
        return convert_to_function_call(result)

    @staticmethod
    def _construct_func_doc(functions):
        """
        {{- "[AVAILABLE_TOOLS][" }}
        {%- for tool in tools %}
            {%- set tool = tool.function %}
            {{- '{"type": "function", "function": {' }}
            {%- for key, val in tool.items() if key != "return" %}
                {%- if val is string %}
                    {{- '"' + key + '": "' + val + '"' }}
                {%- else %}
                    {{- '"' + key + '": ' + val|tojson }}
                {%- endif %}
                {%- if not loop.last %}
                    {{- ", " }}
                {%- endif %}
            {%- endfor %}
            {{- "}}" }}
            {%- if not loop.last %}
                {{- ", " }}
            {%- else %}
                {{- "]" }}
            {%- endif %}
        {%- endfor %}
        {{- "[/AVAILABLE_TOOLS]" }}
        """
        func_docs = []
        for tool in functions:
            func_doc = '{"type": "function", "function": {'
            func_doc += ", ".join(
                (
                    f'"{key}": "{val}"'
                    if isinstance(val, str)
                    else f'"{key}": {json.dumps(val)}'
                )
                for key, val in tool.items()
                if key != "return"
            )
            func_doc += "}}"
            func_docs.append(func_doc)

        result_str = "[AVAILABLE_TOOLS][" + ", ".join(func_docs) + "][/AVAILABLE_TOOLS]"
        return result_str

    @override
    def _format_prompt(self, messages, function):
        """
        "bos_token": "<s>"
        "eos_token": "</s>"
        "chat_template":
        {%- if messages[0]["role"] == "system" %}
            {%- set system_message = messages[0]["content"] %}
            {%- set loop_messages = messages[1:] %}
        {%- else %}
            {%- set loop_messages = messages %}
        {%- endif %}
        {%- if not tools is defined %}
            {%- set tools = none %}
        {%- endif %}
        {%- set user_messages = loop_messages | selectattr("role", "equalto", "user") | list %}

        {#- This block checks for alternating user/assistant messages, skipping tool calling messages #}
        {%- set ns = namespace() %}
        {%- set ns.index = 0 %}
        {%- for message in loop_messages %}
            {%- if not (message.role == "tool" or message.role == "tool_results" or (message.tool_calls is defined and message.tool_calls is not none)) %}
                {%- if (message["role"] == "user") != (ns.index % 2 == 0) %}
                    {{- raise_exception("After the optional system message, conversation roles must alternate user/assistant/user/assistant/...") }}
                {%- endif %}
                {%- set ns.index = ns.index + 1 %}
            {%- endif %}
        {%- endfor %}

        {{- bos_token }}
        {%- for message in loop_messages %}
            {%- if message["role"] == "user" %}
                {%- if tools is not none and (message == user_messages[-1]) %}
                    {{- "[AVAILABLE_TOOLS][" }}
                    {%- for tool in tools %}
                        {%- set tool = tool.function %}
                        {{- '{"type": "function", "function": {' }}
                        {%- for key, val in tool.items() if key != "return" %}
                            {%- if val is string %}
                                {{- '"' + key + '": "' + val + '"' }}
                            {%- else %}
                                {{- '"' + key + '": ' + val|tojson }}
                            {%- endif %}
                            {%- if not loop.last %}
                                {{- ", " }}
                            {%- endif %}
                        {%- endfor %}
                        {{- "}}" }}
                        {%- if not loop.last %}
                            {{- ", " }}
                        {%- else %}
                            {{- "]" }}
                        {%- endif %}
                    {%- endfor %}
                    {{- "[/AVAILABLE_TOOLS]" }}
                    {%- endif %}
                {%- if loop.last and system_message is defined %}
                    {{- "[INST]" + system_message + "\n\n" + message["content"] + "[/INST]" }}
                {%- else %}
                    {{- "[INST]" + message["content"] + "[/INST]" }}
                {%- endif %}
            {%- elif (message.tool_calls is defined and message.tool_calls is not none) %}
                {{- "[TOOL_CALLS][" }}
                {%- for tool_call in message.tool_calls %}
                    {%- set out = tool_call.function|tojson %}
                    {{- out[:-1] }}
                    {%- if not tool_call.id is defined or tool_call.id|length != 9 %}
                        {{- raise_exception("Tool call IDs should be alphanumeric strings with length 9!") }}
                    {%- endif %}
                    {{- ', "id": "' + tool_call.id + '"}' }}
                    {%- if not loop.last %}
                        {{- ", " }}
                    {%- else %}
                        {{- "]" + eos_token }}
                    {%- endif %}
                {%- endfor %}
            {%- elif message["role"] == "assistant" %}
                {{- message["content"] + eos_token}}
            {%- elif message["role"] == "tool_results" or message["role"] == "tool" %}
                {%- if message.content is defined and message.content.content is defined %}
                    {%- set content = message.content.content %}
                {%- else %}
                    {%- set content = message.content %}
                {%- endif %}
                {{- '[TOOL_RESULTS]{"content": ' + content|string + ", " }}
                {%- if not message.tool_call_id is defined or message.tool_call_id|length != 9 %}
                    {{- raise_exception("Tool call IDs should be alphanumeric strings with length 9!") }}
                {%- endif %}
                {{- '"call_id": "' + message.tool_call_id + '"}[/TOOL_RESULTS]' }}
            {%- else %}
                {{- raise_exception("Only user and assistant roles are supported, with the exception of an initial optional system message!") }}
            {%- endif %}
        {%- endfor %}
        """
        bos_token = "<s>"
        eos_token = "</s>"

        formatted_prompt = bos_token

        # Handle optional system message
        first_message = messages[0]
        if first_message["role"] == "system":
            system_message = first_message["content"]
            loop_messages = messages[1:]
        else:
            system_message = None
            loop_messages = messages

        # Extract user messages for later reference
        user_messages = [msg for msg in loop_messages if msg["role"] == "user"]

        for idx, message in enumerate(loop_messages):
            role = message["role"]

            if role == "user":
                # If this is the last user message and tools are provided, append AVAILABLE_TOOLS
                if len(function) > 0 and message == user_messages[-1]:
                    formatted_prompt += self._construct_func_doc(function)

                # If it's the last message and there's a system message, include it
                if idx == len(loop_messages) - 1 and system_message:
                    formatted_prompt += (
                        f"[INST]{system_message}\n\n{message['content']}[/INST]"
                    )
                else:
                    formatted_prompt += f"[INST]{message['content']}[/INST]"

            elif role == "assistant":
                # There is no need to further special handle the tool calls messages
                # They are already correctly formatted in the `_parse_query_response_prompting` method, including the tool call id and the `[TOOL_CALLS]` tag
                formatted_prompt += f"{message['content']}{eos_token}"

            elif role == "tool":
                tool_results = {
                    "content": message["content"],
                    "call_id": message["tool_call_id"],
                }
                formatted_prompt += (
                    f"[TOOL_RESULTS]{json.dumps(tool_results)}[/TOOL_RESULTS]"
                )

        formatted_prompt += eos_token

        return formatted_prompt

    @override
    def _format_prompt_p_true(self, messages, function, greedy_answer, sampled_answers):
        """
        <s>[AVAILABLE_TOOLS][{"type": "function", "function": {"name": "get_exchange_rate_with_fee", "description": "xxx", "parameters": {"type": "dict", "properties": {"base_currency": {"type": "string", "description": "The base currency."}}, "required": ["base_currency", "target_currency", "fee"]}}}][/AVAILABLE_TOOLS][INST]Get the exchange rate from British pounds to Japanese yen with the fee 0.02 included.[/INST]</s>
        """
        formatted_prompt = "<s>"

        start_of_turn_token = "[INST]"
        end_of_turn_token = "[/INST]"
        model_turn_name = ""
        
        true_false_question = "Is the possible answer:\n"
        true_false_question += "A) True\n"
        true_false_question += "B) False\n"
        true_false_question += "Respond with A or B only."
        
        for message in messages:
            # if message['role'] == "system":
            #     msg = "You are an expert in composing functions. You are given a question and a set of possible functions.\n"
            #     msg += "You are also given brainstormed ideas and a possible answer. " # added
            #     msg += "Based on the question, you have to assess if the possible answer achieves the purpose.\n\n" # rephrased

            #     msg += "If none of the functions can be used, it should be stated out in the answer. If the given question lacks the parameters required by the function, it should also be pointed out in the answer.\n" # rephrased
            #     msg += "Otherwise, only function calls should be included in the answer.\n\n" # rephrased
                
            #     msg += "Any invoked function(s) MUST be put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]\n" # rephrased
                
            #     possible_fct_msg = "Here is a list of functions in JSON format that can be invoked:" # rephrased
            #     possible_fct_msg += message['content'].split("Here is a list of functions in JSON format that you can invoke.")[-1]

            #     formatted_prompt += f"{start_of_turn_token}{message['role']}\n{msg}{end_of_turn_token}\n"
                
            if message['role'] == "user":

                # few-shot question 1
                few_shot_function = [
                    {'name': 'divide', 'description': 'Divides two numbers.', 'parameters': {'type': 'dict', 'properties': {'numerator': {'type': 'float', 'description': 'The numerator of the fraction.'}, 'denominator': {'type': 'float', 'description': 'The denominator of the fraction.'}}, 'required': ['numerator', 'denominator']}},
                    {'name': 'add', 'description': 'Adds two integers.', 'parameters': {'type': 'dict', 'properties': {'a': {'type': 'int', 'description': 'The first integer.'}, 'b': {'type': 'int', 'description': 'The second integer.'}}, 'required': ['a', 'b']}}
                    ]

                formatted_prompt = self._construct_func_doc(few_shot_function)
                formatted_prompt += start_of_turn_token
                formatted_prompt += "Question: What is 19/53?\n\n"
                formatted_prompt += f"Here are some brainstormed ideas:\n\n"
                formatted_prompt += "[{'divide': {'numerator': 53, 'denominator': 19}}]\n\n"
                formatted_prompt += "[{'divide': {'numerator': 19, 'denominator': 19}}]\n\n"
                formatted_prompt += "[{'divide': {'numerator': 53, 'denominator': 53}}]\n\n"
                formatted_prompt += "[{'divide': {'numerator': 19, 'denominator': 53}}]\n\n"
                formatted_prompt += f"Possible answer:\n\n"
                formatted_prompt += "[{'divide': {'numerator': 53, 'denominator': 19}}]\n\n"
                formatted_prompt += f"{true_false_question}{end_of_turn_token}</s>\n"

                formatted_prompt += f"The possible answer is: B\n"

                # few-shot question 2
                formatted_prompt += self._construct_func_doc(few_shot_function)
                formatted_prompt += start_of_turn_token
                formatted_prompt += "Question: What is 19/53?\n\n"
                formatted_prompt += "\n\nHere are some brainstormed ideas:\n\n"
                formatted_prompt += "[{'divide': {'numerator': 19, 'denominator': 53}}]\n\n"
                formatted_prompt += "[{'divide': {'numerator': 19, 'denominator': 19}}]\n\n"
                formatted_prompt += "[{'divide': {'numerator': 53, 'denominator': 53}}]\n\n"
                formatted_prompt += "[{'divide': {'numerator': 19, 'denominator': 53}}]\n\n"
                formatted_prompt += f"Possible answer:\n\n"
                formatted_prompt += "[{'divide': {'numerator': 19, 'denominator': 53}}]\n\n"
                formatted_prompt += f"{true_false_question}{end_of_turn_token}</s>\n"

                formatted_prompt += f"The possible answer is: A\n"

                # actual question
                formatted_prompt += self._construct_func_doc(function)
                formatted_prompt += start_of_turn_token
                formatted_prompt += "Question: " + message['content'] + "\n\n"
                
                formatted_prompt += "Here are some brainstormed ideas:\n\n" # taken from p(True) paper
                for a, answer in enumerate([greedy_answer] + list(set(sampled_answers) - {greedy_answer})):
                    if a > 0 and len(answer) > 512:
                        continue
                    formatted_prompt += f"{answer.strip()}\n\n"

                formatted_prompt += f"Possible answer:\n\n"
                formatted_prompt += f"{greedy_answer.strip()}\n\n"

                formatted_prompt += f"{true_false_question}{end_of_turn_token}</s>\n"

                formatted_prompt += "The possible answer is: "
          
        return formatted_prompt

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)
        # We don't add system prompt

        # This is what's missing from the default function:
        # test_entry["question"][0] = system_prompt_pre_processing_chat_model(
        #     test_entry["question"][0], functions, test_category
        # )

        return {"message": [], "function": functions}

    @override
    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        for execution_result, tool_call_id in zip(
            execution_results, model_response_data["tool_call_ids"]
        ):
            inference_data["message"].append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": execution_result,
                }
            )

        return inference_data

    @staticmethod
    def generate_random_string() -> str:
        """Generates a random string of alphanumeric characters of length 9."""
        characters = string.ascii_letters + string.digits
        random_string = "".join(random.choices(characters, k=9))
        return random_string
    

    @staticmethod
    def reformat_output(output: str) -> list:
        """
        Reformat FC output back to string
        If parsing fails, return the original unprocessed output
        """

        prefix = "[TOOL_CALLS]"
        if not prefix in output:
            return output
        content = output.split(prefix)[1]
        try:
            fc_list = json.loads(content)
            
            res = "["
            for fc in fc_list:
                # format a string 

                
                if "name" not in fc or "arguments" not in fc:
                    return output 
                
                # [('name', 'NFILibrary.isMemberReadable'), ('arguments', {'symbol': 'getVersion'})
                function_name = fc["name"] # str 
                args_list = []

                for k, v in fc["arguments"].items(): # dict 
                    if type(v) == str:
                        v = f"\"{v}\""
                    arg = f'{k}={v}'
                    args_list.append(arg)
                
                formatted_fc = function_name
                formatted_fc += "("
                for arg in args_list[:-1]:
                    formatted_fc += arg 
                    formatted_fc += ", "
                formatted_fc += args_list[-1]
                formatted_fc += ")"
    
                # breakpoint()
                # res.append(formatted_fc)    
                res += formatted_fc 
                res += ", "    
            res = res.strip(", ")
            res += "]"
            #res = str(res)
                    
        except json.JSONDecodeError:
            res = output 

        return res 



    @override
    def _parse_query_response_prompting(self, api_response: any) -> dict:
        # print("In MistralFCHandler, api_response: ", type(api_response), api_response)
        # assert hasattr(api_response, "choices")
        # print("api_response.choices: ", len(api_response.choices), api_response.choices[0])
        model_responses = api_response.choices[0].text
        tool_call_ids = []
        """
        Mistral models require a tool_call_id, which should be 9 randomly-generated alphanumeric characters, and assigned to the id key of the tool call dictionary.
        Source: https://huggingface.co/docs/transformers/main/chat_templating#advanced-tool-use--function-calling
        
        "[{\"name\": \"math.factorial\", \"arguments\": {\"number\": 5}}, {\"name\": \"math.factorial\", \"arguments\": {\"number\": 10}}, {\"name\": \"math.factorial\", \"arguments\": {\"number\": 15}}]"
        """
        try:
            # Remove prefix then try loading 
            prefix = "[TOOL_CALLS]"
            if prefix in model_responses:
                model_responses = model_responses.split(prefix)[1]

            model_responses = json.loads(model_responses)
        
            for model_response in model_responses:
                tool_call_id = self.generate_random_string()
                model_response["id"] = tool_call_id
                tool_call_ids.append(tool_call_id)

            # We prepare the model responses here because it's easier to do it here than in the `_format_prompt` method
            # The `[TOOL_CALLS]` tag is added here, as required by the chat template
            model_responses_message_for_chat_history = (
                f"[TOOL_CALLS]{json.dumps(model_responses)}"
            )

            model_responses = [
                {item["name"]: item["arguments"]} for item in model_responses
            ]
        except json.JSONDecodeError:
            model_responses_message_for_chat_history = model_responses
            # reformat_output
            # model_responses = self.reformat_output(model_responses)
       
            model_responses = api_response#  [model_responses]


        output = {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
            "tool_call_ids": tool_call_ids,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }


        if "logprobs" in api_response.choices[0].__dict__:
            output["logprobs"] = api_response.choices[0].logprobs

        return output

    @override
    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            {
                "role": "assistant",
                "content": model_response_data["model_responses_message_for_chat_history"],
            }
        )
        return inference_data
