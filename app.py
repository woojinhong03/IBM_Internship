# -*- coding: utf-8 -*-
import gradio as gr
import pandas as pd
from dotenv import load_dotenv
import os


####################################
# IBM_Cloud 모델 불러오기
####################################

from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

Watsonx_ai_url = 'https://us-south.ml.cloud.ibm.com'

load_dotenv()
IBM_Cloud_API = os.getenv('api_key')
IBM_Project_ID = os.getenv('project_id')

def IBM_LLMS(ibm_model,systemp,DATA):
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

import google.generativeai as genai
Google_Gemini_API = os.getenv('google_api_key')

def GOOGLE_LLM(systemp, DATA):
	SYSTEM_PROMPT = systemp
	genai.configure(api_key=Google_Gemini_API)
	model = genai.GenerativeModel('gemini-1.5-flash')
	response = model.generate_content(SYSTEM_PROMPT + DATA)
	res = {response.text}

	return list(res)[0].strip('\n?')


####################################
# 질문 보내기
####################################

def submit_question(systemp, question, active_models, vote_state):
    if not question.strip():
        return ("질문이 비어있습니다.",)*4 + (vote_state,)

    responses = simulate_model_responses(systemp, question, active_models)
    rA = responses.get("Model_A", "Model_A는 제외됨.")
    rB = responses.get("Model_B", "Model_B는 제외됨.")
    rC = responses.get("Model_C", "Model_C는 제외됨.")
    rD = responses.get("Model_D", "Model_D는 제외됨.")

    return rA, rB, rC, rD, vote_state


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
# 라운드 진행 (자동 확정)
####################################

def next_round_and_auto_finalize(vote_state, active_models):
    model_match = {"Model_A":'gemini-1.5-flash', "Model_B":'ibm/granite-3-8b-instruct',"Model_C":'ibm/granite-3-2-8b-instruct-preview-rc',"Model_D":'meta-llama/llama-3-1-8b-instruct'}
    up_models = [m for m in active_models if vote_state.get(m, "❌") == "⭕"]
    auto_final = False
    final_msg = ""
    final_series = pd.Series([], dtype=object)
    
    if len(up_models) == 0:
        round_msg = "현재 라운드에서 '⭕'된 모델이 없습니다. 모두 탈락."
        new_models = []
    elif len(up_models) == 1:
        only_m = up_models[0]
        round_msg = f"'{only_m}' 한 개만 '⭕' => 자동 최종 확정!"
        final_msg = f"최종 모델은 '{model_match[only_m]}'입니다!"
        final_series = pd.Series([only_m])
        auto_final = True
        new_models = [only_m]
    else:
        round_msg = f"'⭕'된 모델: {up_models}"
        new_models = up_models

    hideA = gr.update(visible=("Model_A" in new_models))
    hideB = gr.update(visible=("Model_B" in new_models))
    hideC = gr.update(visible=("Model_C" in new_models))
    hideD = gr.update(visible=("Model_D" in new_models))
    
    show_restart = gr.update(visible=auto_final)

    return (
        round_msg,
        new_models,
        hideA, hideB, hideC, hideD,
        final_msg,
        final_series,
        auto_final,
        show_restart
    )
    

####################################
# 업/다운 토글
####################################

def toggle_vote(vote_state, model):
    current = vote_state.get(model, "❌")  # 기본값을 "X"로 설정
    new_val = "⭕" if current == "❌" else "❌"  # O/X로 토글
    vote_state[model] = new_val

    emoji = "⭕" if new_val == "⭕" else "❌"  # O/X 이모지로 변경
    label = f"{model} ({emoji})"
    
    return vote_state, label


####################################
# 리더보드 갱신
####################################

def update_score(score_dict, final_series):
    """
    final_series 내 모델들 => +1점
    """
    if not final_series.empty:
        for m in final_series:
            score_dict[m] = score_dict.get(m, 0) + 1

    df = pd.DataFrame(list(score_dict.items()), columns=["Model","Score"])
    df.sort_values(by="Score", ascending=False, inplace=True)
    return score_dict, df

def finalize_models_score(vote_state, active_models, score_dict):
    model_match = {"Model_A":'gemini-1.5-flash', "Model_B":'ibm/granite-3-8b-instruct',"Model_C":'ibm/granite-3-2-8b-instruct-preview-rc',"Model_D":'meta-llama/llama-3-1-8b-instruct'}
    """
    업된 모델들 => 최종 확정 => 점수 반영
    """
    ups = [m for m in active_models if vote_state.get(m, "❌") == "⭕"]
    final_series = pd.Series(ups, dtype=object)

    if len(ups) == 0:
        msg = "업된 모델이 없습니다. 최종선택 불가."
    elif len(ups) == 1:
        msg = f"최종 모델은 '{model_match[ups[0]]}'입니다!"
    else:
        msg = f"최종 모델이 여러 개입니다: {ups}"

    auto_final = (len(ups) >= 1)
    show_restart = gr.update(visible=auto_final)

    # 점수 업데이트
    new_score, new_df = update_score(score_dict, final_series)

    return msg, final_series, auto_final, show_restart, new_score, new_df


####################################
# 처음부터 다시 시작 (점수 유지)
####################################

def restart_all_but_keep_score(active_models, vote_state, final_series):
    """투표 상태만 초기화, 점수 그대로"""
    init_models = ["Model_A","Model_B","Model_C","Model_D"]
    new_vote = {m:"down" for m in init_models}
    new_series = pd.Series([], dtype=object)

    # 모델 열 모두 visible=True 복구
    showA= gr.update(visible=True)
    showB= gr.update(visible=True)
    showC= gr.update(visible=True)
    showD= gr.update(visible=True)
    showE= gr.update(visible=True)
    showF= gr.update(visible=True)

    round_msg = "새로 시작합니다. 질문 입력 후 진행하세요."
    final_msg = ""
    hide_restart = gr.update(visible=False)

    return (
        init_models, new_vote, new_series,    # active_models, vote_state, final_series
        showA, showB, showC, showD, showE, showF,
        round_msg, final_msg,
        False,     # auto_finalized
        hide_restart
    )


####################################
# 메인 App (Tabs: Vote / Leaderboard)
####################################

def build_app():
    with gr.Blocks(css="""
                   .gradio-container {
                        max-width: 1280px;
                        margin: auto;
                        h1{text-align: center;};
                        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
                        border-radius: 10px;
                        padding: 20px;
                    }""") as demo:
        with gr.Tabs():
            with gr.Tab("Main"):
                gr.Markdown("""
            # 🏆 LLM 모델 성능 비교 프로젝트
            
            이 프로젝트는 **LLM 모델 간의 성능을 비교**하고, 
            특히 **한국어 사용의 원활함**을 분석하는 것을 목표로 합니다.
                        
            시중에는 이미 다양한 정량적 지표가 존재하지만, 실제로 우리가 직접 사용해본 결과, 이러한 지표의 순위와는 다른 성능을 보이는 경우가 많았습니다.  
            이에 따라, 단순히 점수로 나타난 지표만이 아니라, 실제 사용자들이 체감하는 만족도를 기반으로 최적의 모델을 선정하는 데 도움을 주고자 하였습니다.  
            또한, 우리가 직접 평가한 결과를 참고하는 것뿐만 아니라, 사용자가 직접 모델을 체험하고 평가함으로써 자신에게 가장 적합한 모델을 찾을 수 있도록 돕는 것이 목표입니다.
            
            🚀 **모델 선정 기준**
            - 파라미터(Parameter) 개수
            - AI 모델 버전
            - 사용 가능한 언어
            - 리소스 사용량
            
            📌 **모델 평가 방식**
            - 사람이 직접 평가하는 **Human Evaluation** 방식
            - 동일한 지문과 질문을 입력하여 비교
            - 입력된 데이터 통일하여 일관성 유지
            
            ✅ **모델 평가 기준**
            - 정보 정확도
            - 문장 완성도
            - 문해력
            - 논리적 근거
            - 안전성 및 편향성 
        """)

# 사용 모델 정보(key model granite 3v)
                gr.Markdown("### 📋 LLM 모델 성능 비교 테이블")

                scoreboard_df1 = gr.Dataframe(
                headers=["모델명", "파라미터 개수", "컨텍스트 크기", "임베딩 크기"],
                datatype=["str","str",'str','str'],
                value=[["Granite-3-8B-Instruct", "8B", "128,000 tokens", "4096"],
            ["Granite-3.2-8B-Instruct-Preview", "8B", "128,000 tokens", "4096"],
            ["Meta-Llama-3-8B", "8B", "128,000 tokens", "4096"],
            ["Gemini 1.5 Flash-8B", "8B", "1,048,576 tokens", "2048"]],
                label="리더보드",
                interactive=False
              )
                scoreboard_df1

                gr.Image("image.png", label="📊 LLM 모델 비교 분석")
            # Vote 탭
            with gr.Tab("Vote"):
                gr.Markdown("# 2025 Winter P-Tech Team1 LLM Test Page \n"
                            "## Guide-line \n"
                            "### - 추가 할 시스템 프롬프트를 입력하세요. 그냥 진행 하셔도 됩니다.\n"
                            "### - 질문을 보낸 후 답변을 기준으로 평가하세요.\n"
                            "### - 업/다운 상태 변환을 통하여 자유롭게 평가 할 수 있습니다.\n"
                            "### - 최종 선택 버튼을 클릭 시 제출됩니다.\n\n\n"
                            "## 모델 테스트")
                with gr.Row():
                    user_question = gr.Textbox(label="질문을 입력하세요", lines=1)
                    systemp = gr.Textbox(label="추가할 시스템 프로젝트를 입력하세요", lines=1)
                submit_btn = gr.Button("질문 보내기")

                with gr.Row():
                    with gr.Column(elem_id="colA") as colA:
                        respA = gr.Textbox(label="Model_A 응답", lines=1, interactive=False)
                        toggleA = gr.Button("Model_A (❌)")
                    with gr.Column(elem_id="colB") as colB:
                        respB = gr.Textbox(label="Model_B 응답", lines=1, interactive=False)
                        toggleB = gr.Button("Model_B (❌)")
                    with gr.Column(elem_id="colC") as colC:
                        respC = gr.Textbox(label="Model_C 응답", lines=1, interactive=False)
                        toggleC = gr.Button("Model_C (❌)")
                    with gr.Column(elem_id="colD") as colD:
                        respD = gr.Textbox(label="Model_D 응답", lines=1, interactive=False)
                        toggleD = gr.Button("Model_D (❌)")
                
                round_msg = gr.Textbox(label="라운드 안내", lines=2, interactive=False)
                round_btn = gr.Button("라운드 한번 더 진행")

                final_msg = gr.Textbox(label="최종 결과 안내", lines=2, interactive=False)
                final_btn = gr.Button("최종 선택")

                restart_btn = gr.Button("처음부터 다시 시작", visible=False)

            # Leaderboard 탭
            
            # import page2
            
            with gr.Tab("Your Leaderboard"):
                gr.Markdown("## 나만의 리더보드 화면 (Scoreboard)")
                # States
                init_models = ["Model_A","Model_B","Model_C","Model_D"]
                active_models_state = gr.State(init_models)
                vote_state = gr.State({m: "❌" for m in init_models})  # 기본값 "X"
                final_series_state = gr.State(pd.Series([], dtype=object))
                auto_finalized_state = gr.State(False)

                score_state = gr.State({m:0 for m in init_models})  # 점수
                scoreboard_df = gr.Dataframe(
                headers=["Model","Score"],
                datatype=["str","number"],
                value=[],
                label="리더보드",
                interactive=False
              )
                scoreboard_df
                gr.Markdown("당신이 테스트한 점수를 여기서 확인하세요. 투표는 Vote 탭에서 진행 가능합니다.")

        # (1) 질문 보내기
        submit_btn.click(
            fn=submit_question,
            inputs=[systemp, user_question, active_models_state, vote_state],
            outputs=[respA, respB, respC, respD, vote_state]
        )

        # (2) 업/다운 토글
        toggleA.click(fn=lambda vs: toggle_vote(vs, "Model_A"), inputs=[vote_state], outputs=[vote_state, toggleA])
        toggleB.click(fn=lambda vs: toggle_vote(vs, "Model_B"), inputs=[vote_state], outputs=[vote_state, toggleB])
        toggleC.click(fn=lambda vs: toggle_vote(vs, "Model_C"), inputs=[vote_state], outputs=[vote_state, toggleC])
        toggleD.click(fn=lambda vs: toggle_vote(vs, "Model_D"), inputs=[vote_state], outputs=[vote_state, toggleD])
        
        
        # (3) 라운드 진행
        round_btn.click(
            fn=next_round_and_auto_finalize,
            inputs=[vote_state, active_models_state],
            outputs=[
                round_msg,
                active_models_state,
                colA, colB, colC, colD,
                final_msg,
                final_series_state,
                auto_finalized_state,
                restart_btn
            ]
        )

        # (4) 최종 선택 -> 점수 갱신
        def finalize_wrapper(vs, am, sc):
            msg, fseries, af, rst_btn, new_sc, new_df = finalize_models_score(vs, am, sc)
            return msg, fseries, af, rst_btn, new_sc, new_df

        final_btn.click(
            fn=finalize_wrapper,
            inputs=[vote_state, active_models_state, score_state],
            outputs=[final_msg, final_series_state, auto_finalized_state, restart_btn, score_state, scoreboard_df]
        )

        # (5) 처음부터 -> 투표만 리셋
        def restart_wrapper(am, vs, fs):
            return restart_all_but_keep_score(am, vs, fs)

        restart_btn.click(
            fn=restart_wrapper,
            inputs=[active_models_state, vote_state, final_series_state],
            outputs=[
                active_models_state, vote_state, final_series_state,
                colA, colB, colC, colD,
                round_msg, final_msg,
                auto_finalized_state,
                restart_btn
            ]
        )

    return demo

if __name__=="__main__":
    app = build_app()
    app.launch(share=True)
    