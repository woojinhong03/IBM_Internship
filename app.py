# -*- coding: utf-8 -*-
import gradio as gr
import pandas as pd
import utils.interaction

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
                
                gr.Markdown("""
                    ## 평가하기  
                    🚀 **정량적 평가 방법**
                    - **ROUGE-1** : 단어(uni-gram) 기반 일치율
                    - **ROUGE-L** : 문장 내에서 가장 긴 일치하는 서열(Longest Common Subsequence, LCS) 기반 점수
                    - **F1 SCORE** : Precision(정밀도)과 Recall(재현율)의 조화 평균으로 계산
                    - **TF-IDF** : 요약 능력 평가
                            
                    📌 **정성적 평가 방법**
                    - **Human Evaluation**
      
                    ✅ **평가 선정 이유**  
                    해당 평가 방식을 선정한 이유는 **한국어에 대한 정량적 평가 지표**가 **영어에 대한 정량적 평가 지표**보다 적어서입니다.  
                    정성적 평가 지표의 경우 정량적 지표에서 나온 점수와 다르게 **사용자가 사용하며 느끼는 느낌**에 더 중점을 두었습니다.  
                    해당 평가를 통해 정량적 평가 데이터로 나온 데이터와 다른 결과가 도출되면서 꼭 **정량적 평가가 정답은 아닌 것**을 설명합니다.  
                    
                """)
                
                with gr.Blocks() as Image_of :
                        gr.Markdown("""## ⚖️ 정량적 평가 지표""")

                        with gr.Row() :
                            gr.Image("img/정량적평가(KeyModel).png", label = "Key Model")
                            gr.Image("img/상위모델포함 정량평가.png", label = "Top Model")

                        gr.Markdown("""
                            🔑 **Key Model**  
                            정량적 평가에선 정답에 한 글자라도 틀리다면 틀린 것으로 표시하기 때문에 점수가 낮게 나오는 것을 확인할 수 있습니다.  
                            **Key Model**인 **Granite v3** 모델과 비교해 큰 차이가 확인되지 않지만, **TF-IDF** 지표에서 **Gemini 1.5 flash** 모델보다 점수가 10% 정도 높게 나오는 것을 알 수 있습니다.  
                            해당 결과로 나오는 이유는 **TF-IDF** 평가 방식이 지문에서 많은 문장을 인용할수록 높은 점수를 주는 방식이기 떄문입니다.  
                            **LLAMA v3.1** 모델의 경우 **TF-IDF** 점수가 타 모델에 비해 유독 높게 나오는데 해당 사유는 앞서 위에서 설명한 **TF-IDF** 방식이  
                            지문에서 가장 많은 문장을 인용한 부분에서 점수를 주기 떄문에 **LLAMA v3.1** 모델의 점수가 높게 나오는 것을 확인할 수 있습니다.  
                            
                            📈 **Top Model**  
                            **Top Model** 지표를 참고하면, Top 모델인 **Mixtral 8x7 v1**의 성능과 **Granite v3**와 약 10% 정도의 성능 차이가 나는 것을 확인할 수 있습니다.  
                            **Gemini 1.5 flash** 모델과 Top 모델을 비교하면 **Gemini 1.5 flash** 모델이 Top 모델보다 **Rouge-1과 Rouge-L** 평가 부분에서  
                            **Gemini 1.5 flash**가 약 3% 정도 높다는 것을 확인할 수 있습니다.  
                            그 외에 사항으론 Top 모델이 **Gemini 1.5 flash** 5% 정도 앞서고 있다는 것을 확인할 수 있습니다.  
                        """)

                # 시각화 자료
                with gr.Blocks() as Image:
                    gr.Markdown("""## ✏️ 정성적 평가 지표""")

                    with gr.Row():
                        gr.Image("img/모델별 총합 수치.png", label = "Total Score")
                        gr.Image("img/정성평가(KeyModel).png", label = "Key Model")
                        gr.Image("img/상위모델포함 정성평가.png", label = "Top model")
                        
                    gr.Markdown("""
                        📊 **Total Score**  
                        위 지표 중 **Total Score 지표**의 경우 각 정성 평가의 항목 점수를 모두 더한 값으로 **Gemini 1.5 flash** 모델이 가장 높은 수치가 나온 것을 보여줍니다.  
                        **Gemini 1.5 flash** 모델의 수치가 높게 나오는 것은 해당 모델이 조정이 된 모델이기 때문입니다.  
                        
                        🔑 **Key Model**  
                        IBM LLM과 타 사 LLM 성능을 비교하기 위해 **Granite v3** 모델을 Key Model로 선정하였습니다.  
                        위 지표를 통해 **Total Score**와 비슷한 내용으로 **Gemini 1.5 flash** 모델이 사용자에게 기본 모델보다 10% 정도 좋다는 의견을 볼 수 있습니다.  
                                
                        📈 **Top Model**  
                        파라미터의 수가 더 많은 모델과 비교 했을 때 조정된 모델인 **Gemini 1.5 flash**이 약 560개의 파라미터 개수를 가진  **Mixtral 8x7 v1** 모델과 성능이 비슷하다는 결론이 나왔습니다.  
                        **Gemini 1.5 flash** 모델은 폐쇄형 모델이기 떄문에 정확한 정보는 알 수 없지만, 해당 결과를 통해 **Gemini 1.5 flash** 모델은 어느정도 조정이 되어 있다고 추측할 수 있을 것 같습니다.  
                    """)

                with gr.Blocks() as endding :
                    gr.Image("img/TEAM 1 MAP.png", label = "Mind Map")



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
            
            

            # States
            

            with gr.Tab("ALL Leaderboard"):
                gr.Markdown("## 🏆전체 리더보드 화면 (Scoreboard)")
                import gspread
                import os
                from dotenv import load_dotenv
                
                init_models = ["Model_A","Model_B","Model_C","Model_D"]
                active_models_state = gr.State(init_models)
                vote_state = gr.State({m: "❌" for m in init_models})  # 기본값 "X"
                final_series_state = gr.State(pd.Series([], dtype=object))
                auto_finalized_state = gr.State(False)

                    
                def fetch_data():
                    return [
                        [worksheet.acell('A2').value, worksheet.acell('B2').value],
                        [worksheet.acell('A3').value, worksheet.acell('B3').value],
                        [worksheet.acell('A4').value, worksheet.acell('B4').value],
                        [worksheet.acell('A5').value, worksheet.acell('B5').value]
                    ]
                
                load_dotenv()
                g_project_id = os.getenv('g_project_id')
                g_private_key_id = os.getenv('g_private_key_id')
                g_private_key = os.getenv('g_private_key')
                g_client_email = os.getenv('g_client_email')
                g_client_id = os.getenv('g_client_id')
                g_client_x509_cert_url = os.getenv('g_client_x509_cert_url')
                
                if g_private_key:
                    g_private_key = g_private_key.replace('\\n', '\n')
                
                data = {
                    "type": "service_account",
                    "project_id": g_project_id,
                    "private_key_id": g_private_key_id,
                    "private_key": g_private_key,
                    "client_email": g_client_email,
                    "client_id": g_client_id,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                    "client_x509_cert_url": g_client_x509_cert_url,
                    "universe_domain": "googleapis.com"
                }
                
                gc = gspread.service_account_from_dict(data)
                spreadsheet_url = "https://docs.google.com/spreadsheets/d/1rj3nwKG1bn6gr4T2hCNEU9ycnENQaat3UbF9KL-PfG8/edit?usp=sharing"
                doc = gc.open_by_url(spreadsheet_url)
                worksheet = doc.worksheet("test1")
                
                def sort(i):
                    return i.sort('Score')
                
                leaderboard = gr.Dataframe(
                    headers=["Model","Score"],
                    datatype=["str","number"],
                    value=fetch_data(),
                    label="리더보드",
                    interactive=True
                )
                sort
                
                
                
                with gr.Row():
                    refresh_btn = gr.Button("데이터 갱신")
                    refresh_btn.click(fn=fetch_data, outputs=leaderboard)
                

                

             
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
        round_btn.click(
            fn=lambda vs: utils.interaction.toggle_vote(vs, "Model_A"), inputs=[vote_state], outputs=[vote_state, toggleA],
        )
        round_btn.click(
            fn=lambda vs: utils.interaction.toggle_vote(vs, "Model_B"), inputs=[vote_state], outputs=[vote_state, toggleB],
        )
        round_btn.click(
            fn=lambda vs: utils.interaction.toggle_vote(vs, "Model_C"), inputs=[vote_state], outputs=[vote_state, toggleC],
        )
        round_btn.click(
            fn=lambda vs: utils.interaction.toggle_vote(vs, "Model_D"), inputs=[vote_state], outputs=[vote_state, toggleD],
        )

        # (4) 최종 선택 -> 점수 갱신
        def finalize_wrapper(vs, am):
            msg, fseries, af, rst_btn, dap = utils.interaction.finalize_models_score(vs, am)

            data = worksheet.get_all_values()
            positions = []
            for row_idx, row in enumerate(data, start=1):  # Google Sheets는 1-based index 사용
                for col_idx, cell in enumerate(row, start=1):
                    if cell == dap:
                        positions.append(row_idx)
            
            po = 'B' + str(positions[0])
            x = worksheet.acell(po).value
            x = int(x) + 1
            worksheet.update([[x]], po)
            
            
            return msg, fseries, af, rst_btn

        final_btn.click(
            fn=finalize_wrapper,
            inputs=[vote_state, active_models_state],
            outputs=[final_msg, final_series_state, auto_finalized_state, restart_btn]
        )
        final_btn.click(fn=fetch_data, outputs=leaderboard)

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
        restart_btn.click(
            fn=lambda vs: utils.interaction.toggle_vote(vs, "Model_A"), inputs=[vote_state], outputs=[vote_state, toggleA],
        )
        restart_btn.click(
            fn=lambda vs: utils.interaction.toggle_vote(vs, "Model_B"), inputs=[vote_state], outputs=[vote_state, toggleB],
        )
        restart_btn.click(
            fn=lambda vs: utils.interaction.toggle_vote(vs, "Model_C"), inputs=[vote_state], outputs=[vote_state, toggleC],
        )
        restart_btn.click(
            fn=lambda vs: utils.interaction.toggle_vote(vs, "Model_D"), inputs=[vote_state], outputs=[vote_state, toggleD],
        )
    return demo

if __name__=="__main__":
    app = build_app()
    app.launch(share=True)