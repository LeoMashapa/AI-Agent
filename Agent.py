import os
import json
import certifi
from google import genai
from google.genai import types
from config import GEMINI_API_KEY
from app.prompts import SYSTEM_PROMPT, TOOL_ROUTER_PROMPT
from app.tools import (
    calculate,
    get_dataset_shape,
    get_dataset_columns,
    summarize_cases_data,
    get_open_case_count,
    get_team_with_most_open_cases,
    get_average_days_open_by_priority,
    filter_cases,
    group_cases,
    get_case_percentage,
    get_case_share_by_group,
    get_oldest_cases,
    get_team_average_days_open_ranking,
    get_priority_average_days_open_ranking,
    check_data_quality
)

# SSL certificate fix
os.environ["SSL_CERT_FILE"] = certifi.where()

client = genai.Client(api_key=GEMINI_API_KEY)


def ask_gemini(prompt, system_prompt, temperature=0.2):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=1024
        )
    )

    return response.text.strip()


def route_tool(user_prompt):
    router_response = ask_gemini(user_prompt, TOOL_ROUTER_PROMPT, temperature=0)

    if router_response == "NO_TOOL":
        return None

    try:
        tool_request = json.loads(router_response)
        return tool_request
    except json.JSONDecodeError:
        return None

def run_tool(tool_request):
    tool_name = tool_request.get("tool")

    if tool_name == "calculate":
        num1 = float(tool_request.get("num1"))
        num2 = float(tool_request.get("num2"))
        operation = tool_request.get("operation")
        return calculate(num1, num2, operation)

    elif tool_name == "get_dataset_shape":
        return get_dataset_shape()

    elif tool_name == "get_dataset_columns":
        return get_dataset_columns()

    elif tool_name == "summarize_cases_data":
        return summarize_cases_data()

    elif tool_name == "get_open_case_count":
        return get_open_case_count()

    elif tool_name == "get_team_with_most_open_cases":
        return get_team_with_most_open_cases()

    elif tool_name == "get_average_days_open_by_priority":
        priority = tool_request.get("priority")
        return get_average_days_open_by_priority(priority)

    elif tool_name == "filter_cases":
        status = tool_request.get("status")
        priority = tool_request.get("priority")
        team = tool_request.get("team")
        metric = tool_request.get("metric", "count")

        return filter_cases(
            status=status,
            priority=priority,
            team=team,
            metric=metric
        )
    
    elif tool_name == "group_cases":
        group_by = tool_request.get("group_by")
        metric = tool_request.get("metric", "count")
        status = tool_request.get("status")
        priority = tool_request.get("priority")
        team = tool_request.get("team")

        return group_cases(
            group_by=group_by,
            metric=metric,
            status=status,
            priority=priority,
            team=team
        )
    

    elif tool_name == "get_case_percentage":
        field = tool_request.get("field")
        value = tool_request.get("value")

        return get_case_percentage(
            field=field,
            value=value
        )

    elif tool_name == "get_case_share_by_group":
        group_by = tool_request.get("group_by")
        status = tool_request.get("status")
        priority = tool_request.get("priority")
        team = tool_request.get("team")

        return get_case_share_by_group(
            group_by=group_by,
            status=status,
            priority=priority,
            team=team
        )
    
    elif tool_name == "get_oldest_cases":
        limit = tool_request.get("limit", 3)
        status = tool_request.get("status")
        priority = tool_request.get("priority")
        team = tool_request.get("team")

        return get_oldest_cases(
            limit=limit,
            status=status,
            priority=priority,
            team=team
        )

    elif tool_name == "get_team_average_days_open_ranking":
        order = tool_request.get("order", "desc")

        return get_team_average_days_open_ranking(
            order=order
        )

    elif tool_name == "get_priority_average_days_open_ranking":
        order = tool_request.get("order", "desc")

        return get_priority_average_days_open_ranking(
            order=order
        )

    elif tool_name == "check_data_quality":
        return check_data_quality()

    else:
        return "Unknown tool"

def ask_ai(prompt):
    try:
        if not GEMINI_API_KEY:
            return "Error: GEMINI_API_KEY was not found. Please check your .env file."

        # Step 1: Ask Gemini if a tool is needed
        tool_request = route_tool(prompt)

        # Step 2: If a tool is needed, run the tool
        if tool_request:
            tool_result = run_tool(tool_request)

            final_prompt = f"""
The user asked:
{prompt}

A Python tool was executed.

Tool request:
{tool_request}

Tool result:
{tool_result}

Your task:
1. Answer the user's question directly
2. Explain what the result means
3. Provide a business interpretation
4. Suggest a useful next step if applicable

Do not just repeat the numbers. Explain the insight clearly.
"""

            return ask_gemini(final_prompt, SYSTEM_PROMPT, temperature=0.3)

        # Step 3: If no tool is needed, answer normally
        return ask_gemini(prompt, SYSTEM_PROMPT, temperature=0.7)

    except Exception as e:
        return f"Error: {str(e)}"
