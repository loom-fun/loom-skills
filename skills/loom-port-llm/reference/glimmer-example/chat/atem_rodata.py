"""Hex rodata for atem.loom (strings shared with chat_probe.py's ATEM model)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chat_probe as c
S = {
 "ch_start": "<|start|>", "ch_message": "<|message|>", "a_eot": "<|eot|>", "ch_close": "<|eom|>",
 "a_system_open": "<|start|>system<|message|>", "a_user_open": "<|start|>user<|message|>",
 "a_assistant_to": "<|start|>assistant to=", "a_self_open": "<|start|>assistant to=self<|message|>",
 "a_answer_open": "<|start|>assistant to=user<|message|>", "a_tool_open": "<|start|>tool ",
 "a_tool_output": '<|message|><tool_output name="', "a_tool_output_mid": '">\n', "a_tool_output_close": "\n</tool_output><|eot|>",
 "a_default_system": "<|start|>system<|message|>" + c.ATEM_DEFAULT_SYSTEM, "a_reasoning": c.ATEM_REASONING,
 "a_tool_head": "\n\n" + c.ATEM_TOOL_HEAD, "a_tool_tail": c.ATEM_TOOL_TAIL,
 "a_ns_open": '{"name": "', "a_ns_close": '", "description": ""}\n', "a_schemas": "// Function schemas",
 "a_fn_open": '\n{"name": "', "a_fn_desc": '", "description": ', "a_fn_params": ', "parameters": ', "a_fn_close": "}",
 "a_quote": '"', "a_null": "null",
 "a_recipients": '\n\n# Valid recipients: "self", ', "a_recipient_close": '.*", ', "a_recipients_end": '"user".<|eot|>',
 "ch_generation": c.ATEM_GENERATION,
 "a_calls_open": '<atem:function_calls>\n<atem:invoke name="', "a_tag_end": '">\n', "a_param_open": '<atem:parameter name="',
 "a_param_mid": '">', "a_param_close": "</atem:parameter>\n", "a_calls_close": "</atem:invoke>\n</atem:function_calls>",
 "a_invoke": '<atem:invoke name="', "a_attr_end": '">', "a_param_end": "</atem:parameter>", "a_invoke_end": "</atem:invoke>",
 "x_call_open": "<tool_call>\n<function=", "x_tag_end": ">\n", "x_param_open": "<parameter=", "x_param_close": "\n</parameter>\n",
 "x_call_close": "</function>\n</tool_call>\n",
 "k_name": "name", "k_tool_call_id": "tool_call_id", "k_tool_calls": "tool_calls", "k_id": "id", "k_arguments": "arguments",
 "k_reasoning": "reasoning_content", "k_description": "description", "k_parameters": "parameters", "k_type": "type", "k_text": "text",
 "t_to_self": "to=self", "t_to_user": "to=user", "t_to": "to=", "a_dot": ".",
}
def rodata():
    return "\n".join(f'global.rodata.def @{k} = bytes("{v.encode().hex()}")\n' for k, v in S.items())
