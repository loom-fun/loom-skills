"""Hex rodata for harmony.loom (strings shared with chat_probe.py's harmony model)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chat_probe as c
_head, _tail = c.HARMONY_SYSTEM.split("{effort}")
S = {
 "h_system_open": "<|start|>system<|message|>" + _head, "h_system_tail": _tail, "h_tools_note": c.HARMONY_TOOLS_NOTE,
 "h_end": "<|end|>", "h_start": "<|start|>", "h_message": "<|message|>", "h_call": "<|call|>",
 "h_developer_open": "<|start|>developer<|message|>", "h_instructions": "# Instructions\n\n", "h_tools_head": "# Tools\n\n",
 "h_blank": "\n\n", "h_user_open": "<|start|>user<|message|>",
 "h_analysis_open": "<|start|>assistant<|channel|>analysis<|message|>", "h_final_open": "<|start|>assistant<|channel|>final<|message|>",
 "h_call_open": "<|start|>assistant to=functions.", "h_call_mid": "<|channel|>commentary json<|message|>",
 "h_result_open": "<|start|>functions.", "h_result_mid": " to=assistant<|channel|>commentary<|message|>",
 "h_generation": c.HARMONY_GENERATION,
 "h_ns_open": "## functions\n\nnamespace functions {\n\n", "h_ns_close": "} // namespace functions",
 "h_comment": "// ", "h_nl": "\n", "h_type_kw": "type ", "h_type_eq": " = ", "h_params_open": "(_: {\n",
 "h_params_close": "}) => any;\n\n", "h_no_params": "() => any;\n\n", "h_optional": "?", "h_colon": ": ", "h_comma_nl": ",\n",
 "h_default": ", // default: ", "h_default_plain": "// default: ", "h_obj_open": "{\n", "h_obj_close": "}", "h_comma": ", ",
 "h_nested": c.HARMONY_NESTED, "h_or": " | ", "h_enum_open": '"', "h_enum_sep": '" | "', "h_null": " | null", "h_array": "[]",
 "h_quote": '"', "h_json_null": "null",
 "t_string": "string", "t_number": "number", "t_integer": "integer", "t_boolean": "boolean", "t_object": "object",
 "t_array": "array", "t_any": "any", "t_any_array": "any[]",
 "k_type": "type", "k_items": "items", "k_enum": "enum", "k_nullable": "nullable", "k_one_of": "oneOf",
 "k_properties": "properties", "k_required": "required", "k_description": "description", "k_default": "default",
 "k_name": "name", "k_parameters": "parameters", "k_tool_calls": "tool_calls", "k_arguments": "arguments",
 "k_reasoning": "reasoning_content", "k_text": "text", "k_developer": "developer",
 # output parsing
 "o_analysis": "<|channel|>analysis", "o_to": "to=", "o_to_assistant": "to=assistant", "o_to_functions": "to=functions.",
 "o_space": " ", "o_special": "<|",
 # hermes text handed to hermes_calls
 "x_open": '<tool_call>\n{"name": "', "x_mid": '", "arguments": ', "x_close": "}\n</tool_call>",
}


def rodata():
    return "\n".join(f'global.rodata.def @{k} = bytes("{v.encode().hex()}")\n' for k, v in S.items())
