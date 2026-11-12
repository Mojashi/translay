# Translay

PC の音（マイク or システム音声）を聞いて、話されている言葉をリアルタイムに文字起こしし、日本語などへ同時に訳して字幕パネルに出す macOS アプリ。すべてローカル（Apple Silicon）で動く。

```
音 ─▶ Whisper large-v3-turbo ─▶ 文ごとに確定 ─▶ PLaMo 3 Translate 0.8B ─▶ 字幕
       エンコーダ: Neural Engine        確定した音は捨てる     GPU / MLX、言語ごとに KV テープ
       デコーダ: GPU / MLX
```

- 聞き取り: 確定していない音だけを約 0.7 秒おきに聞き直し、2 回続けて同じだった文を確定して、その音を手放す。
- 翻訳: 話し途中の文は訳し直し（薄い字）、確定した文は確定訳（濃い字）にする。
- 翻訳先は PLaMo 3 が訳せる 53 言語から、メニューでオン・オフできる（複数可）。

## 使い方

```bash
# モデルを models/ に置く
hf download mlx-community/whisper-large-v3-turbo --local-dir models/whisper-large-v3-turbo
hf download argmaxinc/whisperkit-coreml --include "openai_whisper-large-v3-v20240930/AudioEncoder.mlmodelc/*" \
  --local-dir models/whisperkit
hf download mlx-community/Qwen3-ASR-1.7B-4bit --local-dir models/Qwen3-ASR-1.7B-4bit
hf download pfnet/plamo-3-translate-0.8b-MLX-4bit --local-dir models/plamo-3-translate-0.8b-MLX-4bit

scripts/vendor-fluidaudio.sh        # 比較用ツールの依存（パッケージの解決に必要）
(cd translator && uv sync)          # エンジン（Python / MLX）の環境
scripts/build-app.sh en,ja English  # 最初にオンにする翻訳先、話される言語（auto で自動判定）
open app/build/Translay.app
```

メニューバーの「訳」から:

- 音源（マイク / システム音声）、字幕の開始・停止・消去
- 認識言語（自動判定も可）
- 聞き取り（Whisper large-v3-turbo / Qwen3-ASR 1.7B。切り替えるとエンジンを立ち上げ直す）
- 翻訳先（言語ごとにオン・オフ。エンジンを止めずに次の文から効く）
- 記録（音声と字幕を `~/Library/Application Support/Translay/sessions/` に保存）
- 単語登録（1 行 1 語。聞き取りが寄せられる）

字幕パネルはドラッグで移動できる。システム音声には「画面収録とシステムオーディオ録音」の許可が要る。ログは `~/Library/Logs/Translay/`。

PLaMo 3 Translate は PLaMo Community License（商用利用には別途手続きが必要）。

## 構成

| 場所 | 役割 |
|---|---|
| `app/Sources/Translay` | 音源、エンジンのプロセス管理、字幕板・パネル、メニュー、記録 |
| `translator/translay_mt/engine.py` | エンジン本体: 音を受け取り、聞き取り、訳し、出来事を JSON 行で流す |
| `translator/translay_mt/hearing_whisper.py` | 聞き取り（Whisper）: デコードの決まり、区間の確定、無音・繰り返しの扱い |
| `translator/translay_mt/neural_engine.py` | Whisper のエンコーダを Neural Engine で動かす子プロセス |
| `translator/translay_mt/sentences.py` | 文の切り出し（`uv run python -m translay_mt.sentences_check` で検証） |
| `translator/translay_mt/streaming_translator.py`, `translating.py` | 状態を持つ翻訳器と、複数言語のまとめ訳 |
| `bench/` | 計測スクリプトと結果。Qwen3-ASR や Nemotron など、比較した他の聞き取りエンジンの記録も含む |

聞き取りは 2 つから選べる。Whisper はエンコーダを Neural Engine で動かし、確定した音を捨てながら聞き直す。Qwen3-ASR は GPU だけで動き、確定した文の音を手放しながら、その終わりの数語を足場にして続きを聞き直す（`hearing_sentences.py`。部品は `hearing.py`）。
