# PARC2026 最終モデル技術レポート（2026-10-20版）

## 1. 目的と対象モデル

限られた計算資源でVLAを適応し、推論の動作実行周期と安定性を検証した。対象は **R6 FAST step500（Axis Pi0.5派生）**。本書は10月20日準備版の2種類のZIPに対応する技術記録であり、最終提出枠へのアップロードや本選3出場を示すものではない。

基座はHugging Face公開重み **axisrobotics/Pi0.5-Axis** [1]。保存済み取得provenanceで検証したrevisionは次のとおりである。R6 recipeはこの公開dense重みからの初期化を記録するが、学習実行時に別途revisionを再解決したログは本書では確認していない。

`c342608c489515efa9ba8281400dc648bf3970e1`

checkpointは **r6_dual_l2_fresh500_r1、optimizer step 500**。LoRAをfoldし、配備用の813 tensorのstate dictへ変換した。FASTは本プロジェクトの配備用変換重みの呼称であり、FAST action tokenizerではない。重みは主にFP16、projection層はFP32である。両ZIPは同一の **model_fast.safetensors（9,354,050,544 bytes）** を使用する。提出用checkpointファイルのSHA-256は以下の2行を連結した64桁である。

`e55bf7dcca78b2195993dd7a12b96a4e`
`a13e94761d711c99a82b7411c3f8ceb6`

## 2. 構造・学習データ・学習条件

PaliGemmaの視覚言語backboneとGemma系action expertで構成する。画像2系統、言語指示、ロボット状態を入力し、flow-matchingの速度場を用いた反復サンプリングで連続動作チャンクを生成する。OpenPI系実装を学習に利用し、配備は対応する前後処理・正規化資産とLeRobot系推論実装を組み合わせた。独自の作業は更新範囲の比較、数値安定性の切り分け、推論サーバ接続、実行長の切替と固定初態での対比較である。

データは **physical-intelligence/libero** [2]（LIBERO-Spatial / Object / Goal / 10をLeRobot形式へ変換）、**40 tasks・1,693 episodes・273,465 frames**。native 10 Hz、再標本化なし。取得とmetadataを照合したrevisionは以下である。このR6学習へ自作データは追加していない。後続のnative36データ整備は本checkpointの学習に含まれない。

`a4336d589d589045d1c56423ffdf3b88a0e19b1f`

| 項目 | 実際の設定 |
| --- | --- |
| 学習方式・範囲 | LoRA＋一部head更新。action expert attention/FFNのrank 32、action in/out projection・time MLPを更新。PaliGemma/vision/base凍結。Full FTではない |
| 更新対象 | 18 leaves、24,284,192 parameters（全体の0.7194%） |
| 目的・長さ | valid7_temporal、有効7D actionのみ、予測horizon 10 |
| optimizer | FP32 AdamW、β1/β2=0.9/0.95、ε=1e-8、weight decay=1e-10 |
| 主なhyperparameters | peak LR=1e-6、warmup 100、gradient clip=0.1、EMAなし、fresh QUANTILES正規化 |
| バッチ・期間 | 2 GPU、microbatch 1/GPU、accumulation 32、global batch 64、500 optimizer updates。epochs換算は未記録 |

短い四構成スクリーニングから更新範囲を選び、そのcheckpointを継続せず、基座＋zero-delta LoRAからfresh 500更新を行った。0/1/10/25/50/100/250/500で保存した。全重み学習ではメモリ問題があり更新範囲を絞ったが、少数パラメータの更新自体を性能改善の証拠とはしなかった。

<!-- PAGEBREAK -->

## 3. 推論の工夫と試行錯誤

action chunkを使用し **chunk長10、推論denoising steps 10**。反正規化後の7D actionをキューに入れ、Track1/3は先頭 **8** ステップ、Track2は **4** ステップ実行してから再観測する。**action EMA=0、RTC off、指示aliasなし、境界seam平滑化なし**。モデル読込み前の入力検証、warmupとreset、HTTP入出力を整備した。対応する重み・normalizerと実行器を一組として管理した。

過去の固定80例ではAxis初期48/80、R6 step500は49/80、成功率差の95% bootstrap区間は約−0.0875〜0.1125で確実な改善とはいえなかった。最終の公開LIBERO90対比較では、execute8のチャンク先頭arm6だけを0.9現在＋0.1直前動作とする候補を固定した。15完了対で成功2→2、非対象物体変位の代理失敗12→13、手先RMS jerk平均11.7567→9.9146だった。1例の基盤処理中断は除外した。平均平滑度の改善だけでは採用せず、**候補は不採用**。公開タスクのローカル代理結果であり、新しい公式T2得点ではない。

過去のT1 0.47、T2 0.3457、T3 0は保存済みユーザー報告で、元の提出ZIPに紐づく。今回の再構築ZIPに得点を引き継がない。今回の公式得点、公式L4での時間・VRAM、clean環境での完全再現は未評価である。

## 4. ライセンス・依存コードの開示

**基座重み：Axisの再配布許諾は未確認。** 保存記録の固定revisionの一覧にはREADME/LICENSEがなく、変換configのlicenseはnullである。基盤コードのApache-2.0を派生重みの許諾へ読み替えない。PaliGemmaおよび同梱tokenizerには **Gemma Terms of Use** [3] が関係する。重み全体のライセンス適合・再配布許可が確認済みとは宣言しない。

**学習データ：CC-BY-4.0。** 固定revisionのphysical-intelligence/liberoカード[2]が明記する。出典は同カードおよびLIBERO [4]。LIBEROコードのライセンスからデータ許諾を推定したものではない。

**ZIP内の第三者コード：** vendorのMETADATA／LICENSE／ソースヘッダで確認した範囲は、Apache-2.0（LeRobot、transformers、huggingface_hub）、Apache表記（sentencepiece、tokenizersの厳密版は要確認）、MIT（anyio、h11）、BSD-3-Clause（httpcore、httpx）、LGPL-2.1文書を含むnum2words。Gemma tokenizer資産は上記専用条件で扱う。学習に使用したOpenPIはApache-2.0だが、このZIPへの同梱一覧とは区別する。vendor内の全ファイル・推移的依存の許諾検査と必要noticeの充足は未完了であり、全包適合を主張しない。

## 5. 対応ZIPとSHA-256

本書の対象は下記準備済みZIP。各ハッシュは連続2行を連結する。最終提出枠で実際にアップロード完了したZIPが変わった場合、本欄もそのZIPへ更新する。

**Track1 / Track3：PARC2026_T13_R6_FAST_v2_execute8_20261020.zip**

`38d9317bb0496767c00dbbb7dbff71ca`
`b973c96688ee6b1a30fe43592807571e`

**Track2：PARC2026_T2_R6_FAST_v2_execute4_20261020.zip**

`30c4d880f1b6aaf3746b9bfd979db370`
`6a9e5b8a25835ae5fbf7bc3b587561a1`

推論サーバ基点commit：`23cd1c7824549facb40a948043a1fad7363ef6ce`。Track別実行長の差分は各ZIPの設定に含む。

## 参考資料

[1] https://huggingface.co/axisrobotics/Pi0.5-Axis （上記固定revision）
[2] https://huggingface.co/datasets/physical-intelligence/libero （上記固定revision、CC-BY-4.0）
[3] https://ai.google.dev/gemma/terms
[4] LIBERO, https://arxiv.org/abs/2306.03310

学習・比較の数値は本プロジェクトの保存済みrecipe review、FINAL_FACTS、PAIRED_COMPARISON、PACKAGESの記録に基づく。
