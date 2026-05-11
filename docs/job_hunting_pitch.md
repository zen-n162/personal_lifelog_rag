# Job Hunting Pitch

## 30-Second Pitch

個人の写真・LINE履歴・位置情報を外部APIに送信せず、ローカルVLMとマルチモーダルEmbeddingで統合し、自然文から過去の出来事を検索・要約できるライフログRAGアプリを開発しました。画像理解、検索ランキング、評価、UIレビュー、公開用レポートまで一貫して実装しています。

## 1-Minute Pitch

このプロジェクトでは、写真、LINE履歴、位置情報、通話ログをローカルSQLiteに統合し、Qwen3-VLで画像をcaptionやtagへ変換し、Qwen3-VL-Embeddingで画像検索を行うマルチモーダルRAGを構築しました。ユーザーは「2025年1月は何していた？」「ご飯を食べた写真はいつ？」のように自然文で質問できます。プライバシー保護を最重要視し、外部APIは使わず、公開用レポートでは個人情報を匿名化します。品質管理としてprivate eval、db-check、batch QA、UI reviewを用意し、モデル出力を過信しない設計にしています。

## 3-Minute Pitch

作った理由は、個人の記録が写真、メッセージ、位置情報、通話ログに分散しており、あとから自然に振り返ることが難しいためです。まずローカルDBに各データを保存し、写真にはEXIF/GPS、LINEには時刻と本文、通話には構造化ログ、画像にはOCR/VLM/Embeddingを紐づけました。

Qwen3-VLは画像を説明可能なcaptionやtagへ変換する役割、Qwen3-VL-Embeddingはテキストと画像を同じ検索空間に写す役割です。検索ではEmbeddingだけを信じず、VLM tag、OCR、LINE、event evidence、場所候補、手動レビューを組み合わせてrankingします。VLM-onlyやembedding-onlyは弱い根拠として扱い、「可能性」として回答します。

評価面では、private eval、db-check、release manifest、public/private reportを整備しました。UIでは検索結果やVLM結果を人間が確認し、wrong、hidden、not searchableなどを管理できます。技術的には、ローカルAI、データ設計、検索ランキング、評価設計、プライバシー保護を一つのアプリとして統合した点が強みです。
