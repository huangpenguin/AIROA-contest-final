# PARC2026 v1.2 レポート要件との対応

対象文書は公式v1.2説明資料p22「提出物・提出方法（2/2）レポートについて」とp37の最終提出ZIPとの一致条件。課題の根拠をコース用Quriテンプレートから訂正した。本版は10月20日の準備済みZIPについての技術報告であり、アップロード完了、本選3出場資格、主催者の受理を主張しない。

公式資料：https://docs.google.com/presentation/d/1X9tA-LbxuSNlE2SCW_A-ySdUJ6zMmOv6xA4_mY4aXBo/edit

| 公式要件 | 本版の対応 | 状態 |
| --- | --- | --- |
| p22 対象は本選3出場者のみ | 準備版であることを本文1節に明記 | 出場資格・提出は未確認 |
| p22 11/27 23:59まで（本選3締切後3日以内） | 本表に競技のレポート締切を記録 | コースの11/2締切を本版へ適用しない |
| p22 PDF 2ページ以内、日本語または英語 | 日本語A4 PDF、2ページ | 実ファイル検査済み |
| 学習・推論の工夫と試行錯誤 | 2節の範囲選択・fresh学習、3節のexecuteと不採用候補 | 記載済み |
| モデル名・バージョン | R6 FAST step500、Axis Pi0.5派生、10月20日版 | 記載済み |
| 基座名・重み入手元 | axisrobotics/Pi0.5-Axis、Hugging Face公開配布 | 記載済み |
| 使用重みのversion/revision/commit等 | 完全40桁の取得revision、step500名、サーバcommit | 取得provenanceと学習時の再解決ログを区別 |
| 提出checkpoint hash | model_fast.safetensorsの完全64桁SHA-256 | PACKAGESの両ZIPで同一値を照合 |
| モデル構成・推論方式 | PaliGemma＋action expert、連続動作flow-matching、FP16/headsFP32 | 配備ソースの構成とsample_actions式で確認 |
| Action chunk有無・長さ | chunk10、T1/T3 execute8、T2 execute4、denoising10 | 両ZIPの記録と照合 |
| 学習データ・自作概況 | physical-intelligence/libero固定revision、自作追加なし | R6使用分に限定 |
| 件数・tasks・episodes | 40 tasks、1693 episodes、273465 frames | provenanceとrecipeを照合 |
| 学習方法・steps/epochs | LoRA＋heads、500 updates、epochs換算は未記録 | 記載済み |
| 学習対象parameter・FullFT/LoRA/adapter区分 | rank32、18 leaves、24284192 params、0.7194%、base凍結 | 記載済み |
| 主hyperparameters | AdamW、LR、warmup、β、ε、decay、clip、batch・accumulation | 記載済み |
| 基座ライセンス | Axis再配布許諾未確認、PaliGemma/Gemma条件を開示 | 権利適合自体は未解決 |
| データライセンス | CC-BY-4.0、固定revision公式カードを引用 | データカードと保存READMEで確認 |
| ZIP内第三者コードライセンス | vendorのApache/MIT/BSD/LGPL群と未完了範囲を開示 | 全ファイル・notice充足の承認ではない |
| p37 最終提出枠ZIPとのhash一致 | 2種類のZIP名と完全64桁hashを5節へ記載 | 準備済み包に一致。実際の最終slotへ提出した包が変われば更新必須 |
| 匿名性 | 個人名、メール、homeパス、私有HFアカウント、私的URLなし | 本文・PDF metadata・抽出テキストを検査 |

表記要件を満たす技術報告の準備は完了した。権利許諾の未解決、実提出ZIPの最終照合、主催者の審査は残る。PDF以外の本表・内部SOURCES/OUTPUT/検査ログは補助文書であり、2ページ制限内の本文へ混ぜない。

旧5ページ版は拡張技術資料として元バイトのまま保存し、v1.2適合の提出用本文とは扱わない。
