def convert_tools_to_description(tools: list[dict]) -> str:
    ret = ''
    for i, tool in enumerate(tools):
        assert tool['type'] == 'function'
        fn = tool['function']
        if i > 0:
            ret += '\n'
        ret += f'---- BEGIN FUNCTION #{i + 1}: {fn["name"]} ----\n'
        ret += f'Description: {fn["description"]}\n'

        if 'parameters' in fn:
            ret += 'Parameters:\n'
            properties = fn['parameters'].get('properties', {})
            required_params = set(fn['parameters'].get('required', []))

            for j, (param_name, param_info) in enumerate(properties.items()):
                # Indicate required/optional in parentheses with type
                is_required = param_name in required_params
                param_status = 'required' if is_required else 'optional'
                param_type = param_info.get('type', 'string')

                # Get parameter description
                desc = param_info.get('description', 'No description provided')

                # Handle enum values if present
                if 'enum' in param_info:
                    enum_values = ', '.join(f'`{v}`' for v in param_info['enum'])
                    desc += f'\nAllowed values: [{enum_values}]'

                ret += (
                    f'  ({j + 1}) {param_name} ({param_type}, {param_status}): {desc}\n'
                )
        else:
            ret += 'No parameters are required for this function.\n'

        ret += f'---- END FUNCTION #{i + 1} ----\n'
    return ret


def codeact_tool():
    execute_bash = {
        'type': 'function',
        'function': {
            'name': 'execute_bash',
            'description': """Execute a bash command in the terminal.
* Long running commands: For commands that may run indefinitely, it should be run in the background and the output should be redirected to a file, e.g. command = `python3 app.py > server.log 2>&1 &`.
* One command at a time: You can only execute one bash command at a time. If you need to run multiple commands sequentially, you can use `&&` or `;` to chain them together.
""",
            'parameters': {
                'type': 'object',
                'properties': {
                    'command': {
                        'type': 'string',
                        'description': 'The bash command to execute. Can be empty string to view additional logs when previous exit code is `-1`. Can be `C-c` (Ctrl+C) to interrupt the currently running process. Note: You can only execute one bash command at a time. If you need to run multiple commands sequentially, you can use `&&` or `;` to chain them together.',
                    },
                },
                'required': ['command'],
            },
        },
    }

    str_replace_editor = {
        'type': 'function',
        'function': {
            'name': 'str_replace_editor',
            'description': """Custom editing tool for viewing, creating and editing files in plain-text format
* State is persistent across command calls and discussions with the user
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
* The `undo_edit` command will revert the last edit made to the file at `path`

Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`
""",
            'parameters': {
                'type': 'object',
                'properties': {
                    'command': {
                        'description': 'The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.',
                        'enum': ['view', 'create', 'str_replace', 'insert', 'undo_edit'],
                        'type': 'string',
                    },
                    'path': {
                        'description': 'Absolute path to file or directory, e.g. `/workspace/file.py` or `/workspace`.',
                        'type': 'string',
                    },
                    'file_text': {
                        'description': 'Required parameter of `create` command, with the content of the file to be created.',
                        'type': 'string',
                    },
                    'old_str': {
                        'description': 'Required parameter of `str_replace` command containing the string in `path` to replace.',
                        'type': 'string',
                    },
                    'new_str': {
                        'description': 'Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.',
                        'type': 'string',
                    },
                    'insert_line': {
                        'description': 'Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.',
                        'type': 'integer',
                    },
                    'view_range': {
                        'description': 'Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.',
                        'items': {'type': 'integer'},
                        'type': 'array',
                    },
                },
                'required': ['command', 'path'],
            },
        },
    }

    think = {
        'type': 'function',
        'function': {
            'name': 'think',
            'description': """Use the tool to think about something. It will not obtain new information or make any changes to the repository, but just log the thought. Use it when complex reasoning or brainstorming is needed.

Common use cases:
1. When exploring a repository and discovering the source of a bug, call this tool to brainstorm several unique ways of fixing the bug, and assess which change(s) are likely to be simplest and most effective.
2. After receiving test results, use this tool to brainstorm ways to fix failing tests.
3. When planning a complex refactoring, use this tool to outline different approaches and their tradeoffs.
4. When designing a new feature, use this tool to think through architecture decisions and implementation details.
5. When debugging a complex issue, use this tool to organize your thoughts and hypotheses.

The tool simply logs your thought process for better transparency and does not execute any code or make changes.
""",
            'parameters': {
                'type': 'object',
                'properties': {
                    'content': {'type': 'string', 'description': 'The content of your thought.'},
                },
                'required': ['content'],
            },
        },
    }

    finish = {
        'type': 'function',
        'function': {
            'name': 'finish',
            'description': """Finish the interaction when the task is complete OR if the assistant cannot proceed further with the task.""",
            'parameters': {
                'type': 'object',
                'properties': {
                    'message': {
                        'type': 'string',
                        'description': 'A comprehensive message describing task completion, results achieved, any state changes made, key insights discovered, and other notes.',
                    },
                },
                'required': [],
            },
        },
    }

    return [execute_bash, str_replace_editor, think, finish]


def search_tool():
    search = {
        'type': 'function',
        'function': {
            "name": "search",
            "description": "Performs a web search: supply a string 'query' and optional 'topk'. The tool retrieves the top 'topk' results (default 10) for the query, returning their docid, url, and document content (may be truncated based on token limits).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query string for the search."
                    },
                    "topk": {
                        "type": "integer",
                        "description": "Return the top k pages.",
                    }
                },
                "required": [
                    "query"
                ]
            }
        }
    }
    open_page = {
        'type': 'function',
        'function': {
            'name': 'open_page',
            'description': (
                "Open a page by docid or URL and return the complete content. "
                "Provide either 'docid' or 'url'; if both are provided, prefer 'docid'. "
                "The docid or URL must come from prior search tool results."
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'docid': {
                        'type': 'string',
                        'description': 'Document ID from search results to resolve and fetch.',
                    },
                    'url': {
                        'type': 'string',
                        'description': 'Absolute URL from search results to fetch.',
                    },
                },
                'required': [],
            },
        },
    }
    finish = {
        'type': 'function',
        'function': {
            'name': 'finish',
            'description': """Return the final result when you have a definitive answer or cannot progress further. Provide a concise answer plus a brief, evidence-grounded explanation.""",
            'parameters': {
                'type': 'object',
                'properties': {
                    'answer': {
                        'type': 'string',
                        'description': 'A succinct, final answer.',
                    },
                    'explanation': {
                        'type': 'string',
                        'description': 'A brief explanation for your final answer. For this section only, cite evidence documents inline by placing their docids in square brackets at the end of sentences (e.g., [20]). Do not include citations anywhere else.',
                    },
                    'confidence': {
                        'type': 'string',
                        'description': 'Confidence: your confidence score between 0% and 100% for your answer',
                    },
                },
                'required': ['answer', 'explanation'],
            },
        },
    }
    return [search, open_page, finish]


def math_tool():
    """Tools for a self-contained, rule-scored math episode."""
    think = {
        'type': 'function',
        'function': {
            'name': 'think',
            'description': 'Record one concise intermediate calculation or reasoning step.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'reasoning': {
                        'type': 'string',
                        'description': 'A concise calculation or reasoning step.',
                    },
                },
                'required': ['reasoning'],
            },
        },
    }
    finish = {
        'type': 'function',
        'function': {
            'name': 'finish',
            'description': 'Submit the final numeric answer after completing the calculation.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'answer': {
                        'type': 'string',
                        'description': 'The final numeric answer, for example "42" or "#### 42".',
                    },
                    'explanation': {
                        'type': 'string',
                        'description': 'A brief derivation supporting the answer.',
                    },
                },
                'required': ['answer'],
            },
        },
    }
    return [think, finish]


def branch_tool():
    branch = {
        'type': 'function',
        'function': {
            'name': 'branch',
            'description': """Create a sub-branch to execute a sub-task.""",
            'parameters': {
                'type': 'object',
                'properties': {
                    'description': {
                        'description': 'A concise 3-5 word identifier for the sub-task  (e.g., "Narrow down last 7 overs stats", "Code Exploration", "Test Creation").',
                        'type': 'string'
                    },
                    'prompt': {
                        'description': 'Clear, compact task prompt: state objectives and critical info to preserve in the response. Be brief and informative.',
                        'type': 'string'
                    },
                },
                'required': ['description', 'prompt'],
            },
        },
    }
    return_tool = {
        'type': 'function',
        'function': {
            'name': 'return',
            'description': """Finish the interaction when the sub task is complete OR if the assistant cannot proceed further with the task.""",
            'parameters': {
                'type': 'object',
                'properties': {
                    'message': {
                        'type': 'string',
                        'description': 'A comprehensive message describing sub task outcome.',
                    },
                },
                'required': ['message'],
            },
        },
    }
    return [branch, return_tool]


def graph_tool():
    """Graph operations for ContextGraph-based context management."""
    from .graph_memory_aids import neutral
    active_wording = neutral()
    merge = {
        'type': 'function',
        'function': {
            'name': 'merge',
            'description': (
                "Merge multiple context nodes into a single summary node. "
                "The source nodes are folded (compressed) and replaced by the summary. "
                "Use only distinct ACTIVE IDs listed in the latest Eligible graph-tool node IDs line. "
                "Use this to consolidate findings from multiple branches or search results."
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'node_ids': {
                        'type': 'string',
                        'description': 'Comma-separated distinct ACTIVE node IDs from the latest eligible list (e.g., "n3,n5,n7"). Minimum 2, maximum 6.',
                    },
                    'summary': {
                        'type': 'string',
                        'description': 'A concise summary that captures the key information from all merged nodes.',
                    },
                },
                'required': ['node_ids', 'summary'],
            },
        },
    }
    add_edge = {
        'type': 'function',
        'function': {
            'name': 'add_edge',
            'description': (
                "Create a relationship between two context nodes. "
                "Both endpoints must be distinct ACTIVE IDs from the latest eligible list. "
                "Use this to connect related findings across different branches or search results."
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'source': {
                        'type': 'string',
                        'description': 'Source node ID (e.g., "n3").',
                    },
                    'target': {
                        'type': 'string',
                        'description': 'Target node ID (e.g., "n5").',
                    },
                    'relation': {
                        'type': 'string',
                        'description': 'Relationship type.',
                        'enum': ['causal', 'semantic', 'temporal'],
                    },
                },
                'required': ['source', 'target', 'relation'],
            },
        },
    }
    select_node = {
        'type': 'function',
        'function': {
            'name': 'select',
            'description': (
                "Make a different context node the active node. "
                "The active node determines which part of the context graph is prioritized. "
                "The node must be ACTIVE in the latest eligible list. "
                "Use this to switch the active node between different sub-problems or evidence threads."
            ) if active_wording else (
                "Change the active focus to a different context node. "
                "The active node determines which part of the context graph is prioritized. "
                "The node must be ACTIVE in the latest eligible list. "
                "Use this to shift focus between different sub-problems or evidence threads."
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'node_id': {
                        'type': 'string',
                        'description': ('The node ID to make active (e.g., "n3").' if active_wording
                                        else 'The node ID to focus on (e.g., "n3").'),
                    },
                },
                'required': ['node_id'],
            },
        },
    }
    prune_node = {
        'type': 'function',
        'function': {
            'name': 'prune',
            'description': (
                "Remove a low-value context node from the active graph. "
                "Pruned nodes are excluded from context construction, freeing token budget. "
                "The node must be ACTIVE in the latest eligible list. "
                "Use this to discard irrelevant search results or dead-end investigations."
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'node_id': {
                        'type': 'string',
                        'description': 'The node ID to prune (e.g., "n3"). Cannot prune the root node.',
                    },
                },
                'required': ['node_id'],
            },
        },
    }
    pass_graph = {
        'type': 'function',
        'function': {
            'name': 'pass',
            'description': (
                "Decline a graph operation only when a consolidation checkpoint "
                "explicitly permits pass because the graph is already saturated. "
                "Do not use pass during ordinary research turns."
            ),
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': [],
            },
        },
    }
    return [merge, add_edge, select_node, prune_node, pass_graph]


def alfworld_tool(action_only: bool = False):
    """Tools for ALFWorld text-based household tasks."""
    action = {
        'type': 'function',
        'function': {
            'name': 'action',
            'description': (
                "Execute an action in the household environment. "
                "Choose from the admissible commands listed in the observation. "
                "Common actions: 'go to <location>', 'take <object> from <location>', "
                "'put <object> in/on <location>', 'open <container>', 'close <container>', "
                "'clean <object> with <location>', 'heat <object> with <location>', "
                "'cool <object> with <location>', 'examine <object>', 'inventory', 'look'."
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'command': {
                        'type': 'string',
                        'description': 'The exact action command to execute. Must be one of the admissible commands.',
                    },
                },
                'required': ['command'],
            },
        },
    }
    think = {
        'type': 'function',
        'function': {
            'name': 'think',
            'description': 'Reason about the current situation and plan next steps before acting.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'reasoning': {
                        'type': 'string',
                        'description': 'Your reasoning about what to do next.',
                    },
                },
                'required': ['reasoning'],
            },
        },
    }
    finish = {
        'type': 'function',
        'function': {
            'name': 'finish',
            'description': 'Declare the task complete or give up.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'answer': {
                        'type': 'string',
                        'description': 'Status: "success" if task completed, "failure" if giving up.',
                    },
                },
                'required': ['answer'],
            },
        },
    }
    if action_only:
        return [action]
    return [action, think, finish]


def scienceworld_tool(*, focus_guidance=False):
    """Action tool for the ScienceWorld text simulator."""
    return [{
        'type': 'function',
        'function': {
            'name': 'action',
            'description': (
                "Execute one text command in the ScienceWorld environment. "
                "Use `look around` or `inventory` to inspect state. `focus on <object>` selects "
                "the task target and may terminate the episode with failure if the target is wrong."
                if focus_guidance else
                "Execute one text command in the ScienceWorld environment. "
                "Use `look around`, `inventory`, `focus on <object>`, and the "
                "possible actions/objects reported by observations to plan experiments."
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'command': {
                        'type': 'string',
                        'description': 'The exact ScienceWorld action command to execute.',
                    },
                },
                'required': ['command'],
            },
        },
    }]


TOOL_PROMPT = """
You have access to the following functions:

{description}

If you choose to call a function ONLY reply in the following format with NO suffix:

<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after.
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>
"""

PARALLEL_TOOL_PROMPT = """
You have access to the following functions:

{description}

If you choose to call a function ONLY reply in the following format with NO suffix:

<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after.
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>
"""

#
