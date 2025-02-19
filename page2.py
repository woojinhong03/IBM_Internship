import gradio as gr
import pandas as pd

with gr.Tab("Your Leaderboard"):
                gr.Markdown("## 나만의 리더보드 화면 (Scoreboard)")
                # States
                init_models = ["Model_A","Model_B","Model_C","Model_D"]
                active_models_state = gr.State(init_models)
                # vote_state = gr.State({m:"down" for m in init_models})
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