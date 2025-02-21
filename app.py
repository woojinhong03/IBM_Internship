# -*- coding: utf-8 -*-
import gradio as gr
import pandas as pd
import utils.interaction
import utils.data

####################################
# 메인 App (Tabs: Vote / Leaderboard)
####################################

def build_app() :
    with gr.Blocks(css="""
                   .gradio-container {
                        max-width: 1280px;
                        margin: auto;
                        h1{text-align: center;};
                        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
                        border-radius: 10px;
                        padding: 20px;
                    }""") as demo :
        with gr.Tabs() :
            with gr.Tab("Main") :
                gr.Markdown("""
            # 🏆 LLM 모델 성능 비교 프로젝트
            
             이 프로젝트는 **LLM 모델 간의 성능을 비교**하며,  
            특히 **한국어 사용의 원활함**을 분석하는 것을 목표로 합니다.
                        
             시중에는 다양한 정량적 지표가 존재합니다. 하지만 일상 생활속에서 여러 LLM 모델을 직접 사용해본 결과 **정량적 지표 기반의 순위와 다른 성능을 보이는 경우가 많았습니다.**  
            이에 따라, 단순히 정량적 지표 기반 평가 결과물이 아닌 **실제 사용자들이 체감하는 만족도 기반의 새로운 평가 결과를 만들자.** 라는 결론을 내렸습니다.  
            또한, 저희의 평가 결과와 별개로 사용자가 직접 모델을 체험하고 평가함으로써 **자신에게 가장 적합한 모델**을 찾을 수 있도록 돕는 것이 목표입니다.
            
            **선정 모델**  
            - Granite-3-8B-Instruct  
            - Granite-3.2-8B-Instruct-Preview
            - Meta-Llama-3-8B
            - Gemini 1.5 Flash-8B
            
            🚀 **모델 선정 기준**
            - 파라미터 개수
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
                    value=[
                        ["Granite-3-8B-Instruct", "8B", "128,000 tokens", "4096"],
                        ["Granite-3.2-8B-Instruct-Preview", "8B", "128,000 tokens", "4096"],
                        ["Meta-Llama-3-8B", "8B", "128,000 tokens", "4096"],
                        ["Gemini 1.5 Flash-8B", "8B", "1,048,576 tokens", "2048"]
                    ],
                    label="리더보드",
                    interactive=False
                )
                
                scoreboard_df1

                # 시각화 설명
                gr.Markdown()

                # 시각화 자료
                # gr.Image("img/image.png", label="📊 LLM 모델 비교 분석")


            # Vote 탭
            with gr.Tab("Vote"):
                gr.Markdown(
                    "# 2025 Winter P-Tech Team1 LLM Test Page \n"
                    "## Guide-line \n"
                    "### - 추가 할 시스템 프롬프트를 입력하세요. 그냥 진행 하셔도 됩니다.\n"
                    "### - 질문을 보낸 후 답변을 기준으로 평가하세요.\n"
                    "### - 업/다운 상태 변환을 통하여 자유롭게 평가 할 수 있습니다.\n"
                    "### - 최종 선택 버튼을 클릭 시 제출됩니다.\n\n\n"
                    "## 모델 테스트"
                )
                
                with gr.Row() :
                    user_question = gr.Textbox(label="질문을 입력하세요", lines=1)
                    systemp = gr.Textbox(label="추가할 시스템 프로젝트를 입력하세요", lines=1)

                submit_btn = gr.Button("질문 보내기")

                with gr.Row() :
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
            
            with gr.Tab("Your Leaderboard") :
                gr.Markdown("## 🏆나만의 리더보드 화면 (Scoreboard)")

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
                    interactive=True
                )

                scoreboard_df
                
                test_df = gr.Dataframe(
                    headers=["test"],
                    datatype=["str"],
                    value=[utils.data.dataf],
                    label="test",
                    interactive=True
                )
                
                test_df

                gr.Markdown("당신이 테스트한 점수를 여기서 확인하세요. 투표는 Vote 탭에서 진행 가능합니다.")

        # (1) 질문 보내기
        submit_btn.click(
            fn=utils.interaction.submit_question,
            inputs=[systemp, user_question, active_models_state, vote_state],
            outputs=[respA, respB, respC, respD, vote_state]
        )

        # (2) 업/다운 토글
        toggleA.click(fn=lambda vs: utils.interaction.toggle_vote(vs, "Model_A"), inputs=[vote_state], outputs=[vote_state, toggleA])
        toggleB.click(fn=lambda vs: utils.interaction.toggle_vote(vs, "Model_B"), inputs=[vote_state], outputs=[vote_state, toggleB])
        toggleC.click(fn=lambda vs: utils.interaction.toggle_vote(vs, "Model_C"), inputs=[vote_state], outputs=[vote_state, toggleC])
        toggleD.click(fn=lambda vs: utils.interaction.toggle_vote(vs, "Model_D"), inputs=[vote_state], outputs=[vote_state, toggleD])
        
        
        # (3) 라운드 진행
        round_btn.click(
            fn=utils.interaction.next_round_and_auto_finalize,
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
            msg, fseries, af, rst_btn, new_sc, new_df = utils.interaction.finalize_models_score(vs, am, sc)

            return msg, fseries, af, rst_btn, new_sc, new_df

        final_btn.click(
            fn=finalize_wrapper,
            inputs=[vote_state, active_models_state, score_state],
            outputs=[final_msg, final_series_state, auto_finalized_state, restart_btn, score_state, scoreboard_df]
        )

        # (5) 처음부터 -> 투표만 리셋
        def restart_wrapper(am, vs, fs):
            return utils.interaction.restart_all_but_keep_score(am, vs, fs)

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