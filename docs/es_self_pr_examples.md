# ES Self-PR Examples

## 100 Characters

外部APIを使わず、写真・LINE・位置情報を統合して自然文検索できるローカルマルチモーダルRAGを開発しました。

## 200 Characters

個人の写真・LINE履歴・位置情報をローカル環境で統合し、Qwen3-VLとマルチモーダルEmbeddingを使って自然文で過去の出来事を検索できるRAGアプリを開発しました。評価、UIレビュー、公開用レポートまで整備しました。

## 400 Characters

私は、写真・LINE履歴・位置情報・通話ログを外部APIに送信せずローカルで統合し、自然文から過去の出来事を検索・要約できるマルチモーダルRAGアプリを開発しました。Qwen3-VLで画像captionやtagを生成し、Qwen3-VL-Embeddingで画像検索を行い、OCRやイベント根拠も組み合わせて回答します。VLMの誤推定を前提に、evidence strength、private eval、db-check、UI reviewを用意し、安全性と改善サイクルを重視しました。

## 800 Characters

私は、個人の写真、LINE履歴、位置情報、通話ログをローカル環境で統合し、自然文から過去の出来事を検索・要約できるマルチモーダルRAGアプリを開発しました。背景には、個人の記録が複数の媒体に分散し、あとから「いつ何をしたか」「どんな写真があるか」を調べにくいという課題があります。実装では、SQLiteにデータを集約し、Qwen3-VLで画像captionやscene/activity/food/location cuesを抽出し、Qwen3-VL-Embeddingでtext-to-image検索を行いました。OCR、LINE、GPS由来のevent evidenceも組み合わせ、VLM-onlyやembedding-onlyの結果は弱い根拠として扱う保守的なrankingにしました。また、外部APIを使わないこと、公開用レポートから実写真・生メッセージ・正確な座標を除くこと、UIで誤推定をwrongやhiddenにできることを重視しました。品質管理としてprivate eval、db-check、batch QA、release manifest、公開用HTML生成を整備し、AIモデルを組み込むだけでなく、評価・運用・説明可能性まで含めた開発を行いました。
