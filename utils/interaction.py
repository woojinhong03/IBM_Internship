import utils.models
import gradio as gr
import pandas as pd

####################################
# 질문 보내기
####################################

def submit_question(systemp, question, active_models, vote_state):
    if not question.strip():
        return ("질문이 비어있습니다.",)*4 + (vote_state,)

    responses = utils.models.simulate_model_responses(systemp, question, active_models)
    rA = responses.get("Model_A", "Model_A는 제외됨.")
    rB = responses.get("Model_B", "Model_B는 제외됨.")
    rC = responses.get("Model_C", "Model_C는 제외됨.")
    rD = responses.get("Model_D", "Model_D는 제외됨.")

    return rA, rB, rC, rD, vote_state

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

####################################
# 처음부터 다시 시작 (점수 유지)
####################################

def restart_all_but_keep_score(active_models, vote_state, final_series) :
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
