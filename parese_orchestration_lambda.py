import json
import re
import logging
 
 
RATIONALE_REGEX_LIST = [
    "(.*?)(<function_call>)",
    "(.*?)(<answer>)"
]
RATIONALE_PATTERNS = [re.compile(regex, re.DOTALL) for regex in RATIONALE_REGEX_LIST]
 
RATIONALE_VALUE_REGEX_LIST = [
    "<scratchpad>(.*?)(</scratchpad>)",
    "(.*?)(</scratchpad>)",
    "(<scratchpad>)(.*?)"
]

UNFORMAT_ANSWER = r"(?<=</scratchpad>)(.*)"
UNFORMAT_ANSWER_PATTERN = re.compile(UNFORMAT_ANSWER, re.DOTALL)

RATIONALE_VALUE_PATTERNS = [re.compile(regex, re.DOTALL) for regex in RATIONALE_VALUE_REGEX_LIST]
 
ANSWER_REGEX = r"(?<=<answer>)(.*)"
ANSWER_PATTERN = re.compile(ANSWER_REGEX, re.DOTALL)
 
ANSWER_TAG = "<answer>"
FUNCTION_CALL_TAG = "<function_call>"
 
ASK_USER_FUNCTION_CALL_REGEX = r"(<function_call>user::askuser)(.*)\)"
ASK_USER_FUNCTION_CALL_PATTERN = re.compile(ASK_USER_FUNCTION_CALL_REGEX, re.DOTALL)
 
ASK_USER_FUNCTION_PARAMETER_REGEX = r"(?<=askuser=\")(.*?)\""  
ASK_USER_FUNCTION_PARAMETER_PATTERN = re.compile(ASK_USER_FUNCTION_PARAMETER_REGEX, re.DOTALL)
 
KNOWLEDGE_STORE_SEARCH_ACTION_PREFIX = "x_amz_knowledgebase_"
 
FUNCTION_CALL_REGEX = r"<function_call>(\w+)::(\w+)::(.+)\((.+)\)"
FUNCTION_CALL_PARAMETER_REGEX = r"(\S+)=(\S+)"
FUNCTION_CALL_PARAMETER_PATTERN = re.compile(FUNCTION_CALL_PARAMETER_REGEX)

ANSWER_PART_REGEX = "<answer_part\\s?>(.+?)</answer_part\\s?>"
ANSWER_TEXT_PART_REGEX = "<text\\s?>(.+?)</text\\s?>"  
ANSWER_REFERENCE_PART_REGEX = "<source\\s?>(.+?)</source\\s?>"
ANSWER_PART_PATTERN = re.compile(ANSWER_PART_REGEX, re.DOTALL)
ANSWER_TEXT_PART_PATTERN = re.compile(ANSWER_TEXT_PART_REGEX, re.DOTALL)
ANSWER_REFERENCE_PART_PATTERN = re.compile(ANSWER_REFERENCE_PART_REGEX, re.DOTALL)
 
# You can provide messages to reprompt the LLM in case the LLM output is not in the expected format
MISSING_API_INPUT_FOR_USER_REPROMPT_MESSAGE = "Missing the argument askuser for user::askuser function call. Please try again with the correct argument added"
ASK_USER_FUNCTION_CALL_STRUCTURE_REMPROMPT_MESSAGE = "The function call format is incorrect. The format for function calls to the askuser function must be: <function_call>user::askuser(askuser=\"$ASK_USER_INPUT\")</function_call>."
FUNCTION_CALL_STRUCTURE_REPROMPT_MESSAGE = 'The function call format is incorrect. The format for function calls must be: <function_call>$FUNCTION_NAME($FUNCTION_ARGUMENT_NAME=""$FUNCTION_ARGUMENT_NAME"")</function_call>.'

logger = logging.getLogger()

# This parser lambda is an example of how to parse the LLM output for the default orchestration prompt
def lambda_handler(event, context):
    logger.info("Lambda input: " + str(event))
    print(event)
    
    # Sanitize LLM response
    sanitized_response = sanitize_response(event['invokeModelRawResponse'])
    
    # Parse LLM response for any rationale
    rationale, unformat_answer = parse_rationale(sanitized_response)
    
    # Construct response fields common to all invocation types
    parsed_response = {
        'promptType': "ORCHESTRATION",
        'orchestrationParsedResponse': {
            'rationale': rationale
        }
    }
    
    # Check if there is a final answer
    try:
        final_answer, generated_response_parts = parse_answer(sanitized_response)
    except ValueError as e:
        addRepromptResponse(parsed_response, e)
        return parsed_response
        
    if final_answer:
        parsed_response['orchestrationParsedResponse']['responseDetails'] = {
            'invocationType': 'FINISH',
            'agentFinalResponse': {
                'responseText': final_answer
            }
        }
        
        if generated_response_parts:
            parsed_response['orchestrationParsedResponse']['responseDetails']['agentFinalResponse']['citations'] = {
                'generatedResponseParts': generated_response_parts
            }
       
        logger.info("Final answer parsed response: " + str(parsed_response))
        return parsed_response
    print(f"unformat_answer: {unformat_answer}")
    if unformat_answer:
        parsed_response['orchestrationParsedResponse']['responseDetails'] = {
            'invocationType': 'FINISH',
            'agentFinalResponse': {
                'responseText': unformat_answer
            }
        }
       
        logger.info("Unformat answer parsed response: " + str(parsed_response))
        return parsed_response
    
    # Check if there is an ask user
    try:
        ask_user = parse_ask_user(sanitized_response)
        if ask_user:
            parsed_response['orchestrationParsedResponse']['responseDetails'] = {
                'invocationType': 'ASK_USER',
                'agentAskUser': {
                    'responseText': ask_user
                }
            }
            
            logger.info("Ask user parsed response: " + str(parsed_response))
            return parsed_response
    except ValueError as e:
        addRepromptResponse(parsed_response, e)
        return parsed_response
        
    # Check if there is an agent action

    try:
        parsed_response = parse_function_call(sanitized_response, parsed_response)
        logger.info("Function call parsed response: " + str(parsed_response))
        return parsed_response
    except ValueError as e:
        addRepromptResponse(parsed_response, e)
        return parsed_response
    
    addRepromptResponse(parsed_response, 'Failed to parse the LLM output')
    logger.info(parsed_response)
    return parsed_response
        
    raise Exception("unrecognized prompt type")


def sanitize_response(text):
    pattern = r"(\\n*)"
    text = re.sub(pattern, r"\n", text)
    return text
    
def parse_rationale(sanitized_response):
    # Checks for strings that are not required for orchestration
    rationale_matcher = next((pattern.search(sanitized_response) for pattern in RATIONALE_PATTERNS if pattern.search(sanitized_response)), None)
    
    if rationale_matcher:
        rationale = rationale_matcher.group(1).strip()
        
        # Check if there is a formatted rationale that we can parse from the string
        rationale_value_matcher = next((pattern.search(rationale) for pattern in RATIONALE_VALUE_PATTERNS if pattern.search(rationale)), None)
        if rationale_value_matcher:
            return rationale_value_matcher.group(1).strip(), None
        
        return rationale, None
    else:
        rationale_tag_matcher = next((pattern.search(sanitized_response) for pattern in RATIONALE_VALUE_PATTERNS if pattern.search(sanitized_response)), None)
        if rationale_tag_matcher:
            print(f"raw output!: {sanitized_response}")
            answer_value_matcher = UNFORMAT_ANSWER_PATTERN.search(sanitized_response)
            print(f"unformat answer!: {answer_value_matcher.group(0).strip()}")
            if answer_value_matcher:
                return rationale_tag_matcher.group(1).strip(), answer_value_matcher.group(0).strip()
             
    return None, None
    
def parse_answer(sanitized_llm_response):
    if has_generated_response(sanitized_llm_response):
        return parse_generated_response(sanitized_llm_response) 
 
    answer_match = ANSWER_PATTERN.search(sanitized_llm_response)
    if answer_match and is_answer(sanitized_llm_response):
        return answer_match.group(0).strip(), None
        
    return None, None
  
def is_answer(llm_response):
    return llm_response.rfind(ANSWER_TAG) > llm_response.rfind(FUNCTION_CALL_TAG)
    
def parse_generated_response(sanitized_llm_response):
    results = []
    
    for match in ANSWER_PART_PATTERN.finditer(sanitized_llm_response):
        part = match.group(1).strip()
        
        text_match = ANSWER_TEXT_PART_PATTERN.search(part)
        if not text_match:
            raise ValueError("Could not parse generated response")
        
        text = text_match.group(1).strip()        
        references = parse_references(sanitized_llm_response, part)
        results.append((text, references))
    
    final_response = " ".join([r[0] for r in results])
    
    generated_response_parts = []
    for text, references in results:
        generatedResponsePart = {
            'text': text, 
            'references': references
        }
        generated_response_parts.append(generatedResponsePart)
        
    return final_response, generated_response_parts

    
def has_generated_response(raw_response):
    return ANSWER_PART_PATTERN.search(raw_response) is not None
 
def parse_references(raw_response, answer_part):
    references = []
    for match in ANSWER_REFERENCE_PART_PATTERN.finditer(answer_part):
        reference = match.group(1).strip()
        references.append({'sourceId': reference})
    return references
    
def parse_ask_user(sanitized_llm_response):
    ask_user_matcher = ASK_USER_FUNCTION_CALL_PATTERN.search(sanitized_llm_response)
    if ask_user_matcher:
        try:
            ask_user = ask_user_matcher.group(2).strip()
            ask_user_question_matcher = ASK_USER_FUNCTION_PARAMETER_PATTERN.search(ask_user)
            if ask_user_question_matcher:
                return ask_user_question_matcher.group(1).strip()
            raise ValueError(MISSING_API_INPUT_FOR_USER_REPROMPT_MESSAGE)
        except ValueError as ex:
            raise ex
        except Exception as ex:
            raise Exception(ASK_USER_FUNCTION_CALL_STRUCTURE_REMPROMPT_MESSAGE)
        
    return None
 
def parse_function_call(sanitized_response, parsed_response):
    print("Start parse fncation call!")
    match = re.search(FUNCTION_CALL_REGEX, sanitized_response)
    if not match:
        raise ValueError(FUNCTION_CALL_STRUCTURE_REPROMPT_MESSAGE)
    
    verb, resource_name, function = match.group(1), match.group(2), match.group(3)
    parameters = {}
    # parameters = {"buyer_tax_number": {
    #         "value": "91440300MA5FAE9E4P"
    #       },
    #       "buyer_company_name": {
    #         "value": "华韵公司"
    #       },
    #       "user_id": {
    #         "value": "00001"
    #       },
    #       "product_detail": {
    #         "value": '[\"{\"name\":\"小麦\",\"code\":\"1010101020000000000\",\"money\":\"9000\"}\"]'
    #       }
    #      }
    # print(type(parameters))
    # print(type(parameters["product_detail"]["value"]))
    # print(parameters["product_detail"]["value"])
    parameter_list = FUNCTION_CALL_PARAMETER_PATTERN.findall(match.group(4))
    for arg in parameter_list:
        print(arg)
        value = json.loads((arg[1].strip().strip(",")))
        print(type(value))
        if isinstance(value, list) and isinstance(value[0], dict):
            print(type(value), type(value[0]))
        
            parameters[arg[0].strip()] = {'value': json.dumps(value, ensure_ascii=False)}
        parameters[arg[0].strip()] = {'value': str(value)}
        print(type(parameters[arg[0].strip()]["value"]))
        # parameters[arg[0].strip()] = {'value': eval(arg[1].strip().strip(","))}
        # parameters.append({'name':arg[0].strip(),'value': eval(arg[1].strip().strip(","))})
    # print(parameters)
    parsed_response['orchestrationParsedResponse']['responseDetails'] = {}
        
    # Function calls can either invoke an action group or a knowledge base.
    # Mapping to the correct variable names accordingly
    if resource_name.lower().startswith(KNOWLEDGE_STORE_SEARCH_ACTION_PREFIX):
        parsed_response['orchestrationParsedResponse']['responseDetails']['invocationType'] = 'KNOWLEDGE_BASE'
        parsed_response['orchestrationParsedResponse']['responseDetails']['agentKnowledgeBase'] = {
            'searchQuery': parameters['searchQuery'],
            'knowledgeBaseId': resource_name.replace(KNOWLEDGE_STORE_SEARCH_ACTION_PREFIX, '')
        }
        
        return parsed_response
    
    parsed_response['orchestrationParsedResponse']['responseDetails']['invocationType'] = 'ACTION_GROUP'
    parsed_response['orchestrationParsedResponse']['responseDetails']['actionGroupInvocation'] = {
        "verb": verb, 
        "actionGroupName": resource_name,
        "apiName": function,
        "actionGroupInput": parameters
    }
    
    return parsed_response
    
def addRepromptResponse(parsed_response, error):
    error_message = str(error)
    logger.warn(error_message)
    
    parsed_response['orchestrationParsedResponse']['parsingErrorDetails'] = {
        'repromptResponse': error_message
    }