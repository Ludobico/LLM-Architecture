import inspect
import json
import re
from typing import Any, Callable, Dict, Optional, Tuple, Union, get_args, get_origin, get_type_hints

BASIC_TYPES = (int, float, str, bool, Any, type(None), ...)

description_re = re.compile(r"^(.*?)[\n\s]*(Args:|Returns:|Raises:|\Z)", re.DOTALL)
# Extracts the Args: block from the docstring
args_re = re.compile(r"\n\s*Args:\n\s*(.*?)[\n\s]*(Returns:|Raises:|\Z)", re.DOTALL)
# Splits the Args: block into individual arguments
args_split_re = re.compile(
    r"""
(?:^|\n)  # Match the start of the args block, or a newline
\s*(\w+):\s*  # Capture the argument name and strip spacing
(.*?)\s*  # Capture the argument description, which can span multiple lines, and strip trailing spacing
(?=\n\s*\w+:|\Z)  # Stop when you hit the next argument or the end of the block
""",
    re.DOTALL | re.VERBOSE,
)
# Extracts the Returns: block from the docstring, if present. Note that most chat templates ignore the return type/doc!
returns_re = re.compile(r"\n\s*Returns:\n\s*(.*?)[\n\s]*(Raises:|\Z)", re.DOTALL)

class TypeHintParsingException(Exception):
    pass

class DocstringParsingException(Exception):
    pass

def _get_json_schema_type(param_type : str) -> Dict[str, str]:
    type_mapping = {
        int : {"type" : "integer"},
        float : {"type" : "number"},
        str : {"type" : "string"},
        bool : {"type" : "boolean"},
        Any : {},
    }

    return type_mapping.get(param_type, {"type" : "object"})

def _parse_type_hint(hint : str) -> Dict:
    origin = get_origin(hint)
    args = get_args(hint)

    if origin is None:
        try:
            return _get_json_schema_type(hint)
        except KeyError:
            raise TypeHintParsingException("Couldn't parse this type hint, likely due to a custom class or object : ", hint)
    
    elif origin is Union:
        subtypes = [_parse_type_hint(t) for t in args if t != type(None)]

        if len(subtypes) == 1:
            return_dict = subtypes[0]
        
        elif all(isinstance(subtypes["type"], str) for subtype in subtypes):
            return_dict = {"type" : [subtypes["type"] for subtype in subtypes]}
        
        else:
            return_dict = {"anyOf" : subtypes}
        
        if type(None) in args:
            return_dict["nullable"] = True
        return return_dict
    
    elif origin is list:
        if not args:
            return {"type" : "array"}
        else:
            return {"type" : "array", "items" : _parse_type_hint(args[0])}
        
    elif origin is tuple:
        if not args:
            return {"type" : "array"}
        if len(args) == 1:
            raise TypeHintParsingException(
                f"The type hint {str(hint).replace("typing.", '')} is a Tuple with a single element. which"
                "We do not automaticlly convert to JSON schema as it is rarely necessary. If this input can contain"
                "more then one element, we recommend"
                "using a List[] type instead, or if it really is a single element. remove the Tuple[] wrapper and just"
                "pass the element directly."
            )
        
        if ... in args:
            raise TypeHintParsingException(
                "Conversaion of '...' is not supported in Tuple type hint. use List[] types for variable-length inputs instead."
            )
        
        return {"type" : "array", "prefixItems" : [_parse_type_hint(t) for t in args]}
    
    elif origin is dict:
        out = {"type" : "object"}
        if len(args) == 2:
            out["additionalProperties"] = _parse_type_hint(args[1])
        return out
    
    raise TypeHintParsingException("Couldn't parse this type hint, likely due to a custom class or object : ", hint)

def _convert_type_hints_to_json_schema(func : Callable) -> Dict:
    type_hints = get_type_hints(func)
    signature = inspect.signature(func)
    required = []

    for param_name, param in signature.parameters.items():
        if param.annotation == inspect.Parameter.empty:
            raise TypeHintParsingException(f"Argument {param_name} is missing a type hint in function {func.__name__}")
        if param.default == inspect.Parameter.empty:
            required.append(param_name)
    
    properties = {}
    for param_name, param_type in type_hints.items():
        properties[param_name] = _parse_type_hint(param_type)

    schema = {"type" : "object", "properties" : properties}
    if required:
        schema['required'] = required
    
    return schema


def parse_google_format_docstring(docstring : str) -> Tuple[Optional[str], Optional[Dict], Optional[str]]:
    description_math = description_re.search(docstring)
    args_match = args_re.search(docstring)
    returns_match = returns_re.search(docstring)

    description = description_math.group(1).strip() if description_math else None
    docstring_args = args_match.group(1).strip() if args_match else None
    returns = returns_match.group(1).strip() if returns_match else None

    if docstring_args is not None:
        docstring_args = '\n'.join([line for line in docstring_args.split('\n') if line.strip()])
        matches = args_split_re.findall(docstring_args)
        args_dict = {match[0] : re.sub(r"\s*\n+\s*", " ", match[1].strip()) for match in matches}
    else:
        args_dict = {}
    
    return description, args_dict, returns


def get_json_schema(func : Callable) -> Dict:
    doc = inspect.getdoc(func)
    if not doc:
        raise DocstringParsingException(
            f"Cannot generate JSON Schema for {func.__name__} because it has no docstring!"
        )
    
    doc = doc.strip()
    main_doc, param_descriptions, return_doc = parse_google_format_docstring(doc)

    json_schema = _convert_type_hints_to_json_schema(func)
    if (return_dict := json_schema['properties'].pop("return", None)) is not None:
        if return_doc is not None:
            return_dict['description'] = return_doc
    for arg, schema in json_schema['properties'].items():
        if arg not in param_descriptions:
            raise DocstringParsingException(
                f"Cannot generate JSON Schema for {func.__name__} because it has no description for argument {arg}!"
            )
        
        desc = param_descriptions[arg]
        enum_choices = re.search(r"\(choices:\s*(.*?)\)\s*$", desc, flags=re.IGNORECASE)
        if enum_choices:
            schema['enum'] = [c.strip() for c in json.load(enum_choices.group(1))]
            desc = enum_choices.string[: enum_choices.start()].strip()
        schema['description'] = desc

    output = {"name" : func.__name__, "description" : main_doc, "parameters" : json_schema}

    if return_dict is not None:
        output['return'] = return_dict
    return {"type" : "function", "function" : output}