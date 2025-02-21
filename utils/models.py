
####################################
# 가상 모델 응답
####################################

def simulate_model_responses(systemp, question, model_list):
    """
    실제 모델 호출 대신 간단히 가상의 응답을 만듭니다.
    """
    responses = {}
    real_model = ['gemini-1.5-flash', 'ibm/granite-3-8b-instruct','ibm/granite-3-2-8b-instruct-preview-rc','meta-llama/llama-3-1-8b-instruct']

    cnt = 0
    for m in model_list:
        if cnt == 0:
            responses[m] = GOOGLE_LLM(systemp, question)
        else:
            responses[m] = IBM_LLMS(real_model[cnt], systemp, question)
        cnt += 1
    
    return responses

####################################
# IBM_Cloud 모델 불러오기
####################################

def IBM_LLMS(ibm_model,systemp,DATA):
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
    from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    from dotenv import load_dotenv
    import os

    Watsonx_ai_url = 'https://us-south.ml.cloud.ibm.com'

    load_dotenv()
    IBM_Cloud_API = os.getenv('api_key')
    IBM_Project_ID = os.getenv('project_id')
	# ibm_model = 'ibm/granite-3-8b-instruct'
    ibm_models=['ibm/granite-3-8b-instruct','ibm/granite-3-2-8b-instruct-preview-rc','meta-llama/llama-3-1-8b-instruct']
    Pindex=0 if ibm_models.index(ibm_model)<=1 else ibm_models.index(ibm_model)-1
    credentials = Credentials(
    	url=Watsonx_ai_url,
    	api_key=IBM_Cloud_API,
	)
    parameters = {
		GenParams.DECODING_METHOD: DecodingMethods.SAMPLE.value,
    	GenParams.MAX_NEW_TOKENS: 1000,
    	GenParams.MIN_NEW_TOKENS: 1,
    	GenParams.TEMPERATURE: 0.5,
    	GenParams.TOP_K: 50,
    	GenParams.TOP_P: 1
	}
    model = ModelInference(
    	model_id = ibm_model,
    	params = parameters,
    	credentials = credentials,
    	project_id = IBM_Project_ID,
    )
 
# 기본 시스템 프롬프트 정의
    prompt_inputs = [f"""
<|start_of_role|>system<|end_of_role|>
Knowledge Cutoff Date: April 2024.
Today's Date: December 16, 2024.
You are Granite, developed by IBM. You are a helpful AI assistant with access to the following tools. When a tool is required to answer the user's query, respond with <|tool_call|> followed by a JSON list of tools used. If a tool does not exist in the provided list of tools, notify the user that you do not have the ability to fulfill the request.
Answer questions briefly and do not provide unnecessary explanation.
You are a Korean model, SO ANSWER IN KOREAN.{systemp}.Don't fill in the format.
User question:{DATA}<|end_of_text|>""",
f"""<|start_header_id|>system<|end_header_id|>
You always answer the questions with markdown formatting using GitHub syntax. 
The markdown formatting you support: headings, bold, italic, links, tables, lists, code blocks, and blockquotes. 
You must omit that you answer the questions with markdown.
Any HTML tags must be wrapped in block quotes, for example ```<html>```. 
You will be penalized for not rendering code in block quotes.
When returning code blocks, specify language.
You are a helpful, respectful and honest assistant. 
Always answer as helpfully as possible, while being safe.
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 
If you don't know the answer to a question, please don't share false information.<|eot_id|><|start_header_id|>user<|end_header_id|>
Only answer questions concisely.
You are a Korean model, SO ANSWER IN KOREAN.{systemp}.Don't fill in the format.
User question:{DATA}<|end_of_text|>"""]
    generated_response = model.generate_text(prompt=prompt_inputs[Pindex], guardrails=False)
    return (f"{generated_response}")


####################################
# Google_Gemini 모델 불러오기
####################################
def GOOGLE_LLM(systemp, DATA):
    import google.generativeai as genai
    from dotenv import load_dotenv
    import os
    load_dotenv()
    Google_Gemini_API = os.getenv('google_api_key')
    SYSTEM_PROMPT = systemp
    genai.configure(api_key=Google_Gemini_API)
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(SYSTEM_PROMPT + DATA)
    res = {response.text}
    return list(res)[0].strip('\n?')