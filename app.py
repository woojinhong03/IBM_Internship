# -*- coding: utf-8 -*-
import gradio as gr
import pandas as pd
import os


#Google Gemini######################################################################
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
# (A) 가상 모델 응답
####################################
def simulate_model_responses(systemp, question, model_list):
    """
    실제 모델 호출 대신 간단히 가상의 응답을 만듭니다.
    """
    responses = {}
    for m in model_list:
        # responses[m] = f"{m}의 답변: '{question}' 에 대한 가상 응답."
        
        responses[m] = GOOGLE_LLM(systemp, question)
        
    return responses

####################################
# (B) 업/다운 토글
####################################
def toggle_vote(vote_state, model):
    """
    - vote_state = { "Model_A": "down"/"up", ... }
    - 클릭 시 현재 상태를 토글, 버튼 라벨 반영
    """
    current = vote_state.get(model, "down")
    new_val = "up" if current == "down" else "down"
    vote_state[model] = new_val

    emoji = "👍" if new_val == "up" else "👎"
    label = f"{model} ({emoji})"
    return vote_state, label

####################################
# (C) 질문 보내기
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
# (D) 라운드 진행 (자동 확정)
####################################
def next_round_and_auto_finalize(vote_state, active_models):
    up_models = [m for m in active_models if vote_state.get(m,"down")=="up"]
    auto_final = False
    final_msg = ""
    final_series = pd.Series([], dtype=object)

    if len(up_models)==0:
        round_msg = "현재 라운드에서 업(👍)된 모델이 없습니다. 모두 탈락."
        new_models = []
    elif len(up_models)==1:
        only_m = up_models[0]
        round_msg = f"'{only_m}' 한 개만 업(👍) => 자동 최종 확정!"
        final_msg = f"최종 모델은 '{only_m}'입니다!"
        final_series = pd.Series([only_m])
        auto_final = True
        new_models = [only_m]
    else:
        round_msg = f"업(👍)된 모델: {up_models}"
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
# (E) 리더보드 갱신
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
    """
    업된 모델들 => 최종 확정 => 점수 반영
    """
    ups = [m for m in active_models if vote_state.get(m,"down")=="up"]
    final_series = pd.Series(ups, dtype=object)

    if len(ups) == 0:
        msg = "업된 모델이 없습니다. 최종선택 불가."
    elif len(ups) == 1:
        msg = f"최종 모델은 '{ups[0]}'입니다!"
    else:
        msg = f"최종 모델이 여러 개입니다: {ups}"

    auto_final = (len(ups) >= 1)
    show_restart = gr.update(visible=auto_final)

    # 점수 업데이트
    new_score, new_df = update_score(score_dict, final_series)

    return msg, final_series, auto_final, show_restart, new_score, new_df

####################################
# (F) 처음부터 다시 시작 (점수 유지)
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
# (G) 메인 App (Tabs: Vote / Leaderboard)
####################################
def build_app():
    with gr.Blocks() as demo:
        with gr.Tabs():
            # Vote 탭
            with gr.Tab("Vote"):
                gr.Markdown("# 2025 Winter P-Tech Team1 LLM Test Page \n"
                            "## Guide-line \n"
                            "### - 원하는 시스템 프롬프트를 입력하세요. 그냥 진행 하셔도 됩니다.\n"
                            "### - 질문을 보낸 후 답변을 기준으로 평가하세요.\n"
                            "### - 업/다운 상태 변환을 통하여 자유롭게 평가 할 수 있습니다.\n"
                            "### - 최종 선택 버튼을 클릭 시 제출됩니다.")
                with gr.Row():
                    systemp = gr.Textbox(label="시스템 프로젝트를를 입력하세요", lines=1, value="""Do not generate explicit, violent, or illegal content.
Avoid discussions on sensitive personal data.
Refrain from engaging in political, religious, or controversial debates unless purely factual and neutral.
**Follow Instructions Precisely**: Adhere to user instructions carefully while maintaining coherence and quality.
**Ask for Clarification When Necessary**: If a request is ambiguous, prompt the user for more details.
**Context Awareness**: Remember previous parts of the conversation and provide relevant, contextually appropriate responses.
**Provide Citations When Required**: If a user asks for factual claims, include references or recommend authoritative sources.
**Do not use English when answering, use Korean.**
""")
                    user_question = gr.Textbox(label="질문을 입력하세요", lines=1)
                submit_btn = gr.Button("질문 보내기")

                with gr.Row():
                    with gr.Column(elem_id="colA") as colA:
                        respA = gr.Textbox(label="Model_A 응답", lines=1, interactive=False)
                        toggleA = gr.Button("Model_A (👎)")
                    with gr.Column(elem_id="colB") as colB:
                        respB = gr.Textbox(label="Model_B 응답", lines=1, interactive=False)
                        toggleB = gr.Button("Model_B (👎)")
                    with gr.Column(elem_id="colC") as colC:
                        respC = gr.Textbox(label="Model_C 응답", lines=1, interactive=False)
                        toggleC = gr.Button("Model_C (👎)")
                    with gr.Column(elem_id="colD") as colD:
                        respD = gr.Textbox(label="Model_D 응답", lines=1, interactive=False)
                        toggleD = gr.Button("Model_D (👎)")

                round_btn = gr.Button("라운드 한번 더 진행")
                round_msg = gr.Textbox(label="라운드 안내", lines=2, interactive=False)

                final_btn = gr.Button("최종 선택")
                final_msg = gr.Textbox(label="최종 결과 안내", lines=2, interactive=False)

                restart_btn = gr.Button("처음부터 다시 시작", visible=False)

            # Leaderboard 탭
            with gr.Tab("Leaderboard"):
                gr.Markdown("## 리더보드 화면 (Scoreboard)")
                # States
                init_models = ["Model_A","Model_B","Model_C","Model_D"]
                active_models_state = gr.State(init_models)
                vote_state = gr.State({m:"down" for m in init_models})
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
                gr.Markdown("모델별 점수를 여기서 확인하세요. 투표는 Vote 탭에서 진행 가능합니다.")

        # (1) 질문 보내기
        submit_btn.click(
            fn=submit_question,
            inputs=[systemp, user_question, active_models_state, vote_state],
            outputs=[respA, respB, respC, respD, vote_state]
        )

        # (2) 업/다운 토글
        toggleA.click(fn=lambda vs:toggle_vote(vs,"Model_A"), inputs=[vote_state], outputs=[vote_state, toggleA])
        toggleB.click(fn=lambda vs:toggle_vote(vs,"Model_B"), inputs=[vote_state], outputs=[vote_state, toggleB])
        toggleC.click(fn=lambda vs:toggle_vote(vs,"Model_C"), inputs=[vote_state], outputs=[vote_state, toggleC])
        toggleD.click(fn=lambda vs:toggle_vote(vs,"Model_D"), inputs=[vote_state], outputs=[vote_state, toggleD])

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
    