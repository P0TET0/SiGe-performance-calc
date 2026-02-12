# SiGe-performance-calc

Windowsローカル環境で動作する、SiGe 熱電性能（ZT）の簡易シミュレーターです。  
`streamlit` で1画面UIを提供し、組成比 `y` と `N_D` を選んで ZT グラフを表示し、PNGで保存できます。

## 前提
- Windows
- オフライン実行（ローカルのみ）
- Python 3.10+

## セットアップ

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## 使い方
1. `組成比` を選択
2. `N_D` を選択
3. `計算して表示` を押す
4. グラフを確認し、`PNGをダウンロード` を押す

PNGファイル名には組成比と `N_D` が含まれます。

## データファイル
- 必須 `N_D` 候補:
  - `C:\Users\miots\ruruproject\SiGe-performance\SiGe-performance-calc\data\N_D_values.pkl`
- 自動利用（存在する場合）:
  - `data/T_range.pkl`
  - `data/xi_F_vals.pkl`

`T_range.pkl` と `xi_F_vals.pkl` が有効なときは `ZT vs Temperature [K]` を表示します。  
利用できない場合はフォールバックとして `ZT vs xi_F` を表示します。

## N_D_values.pkl が見つからない場合
1. まず上記の必須パスにファイルがあるか確認
2. プロジェクト配下の `data/N_D_values.pkl` を配置（フォールバック）
3. 必要なら `settings.py` のパス設定を調整

## 構成
- `app.py`: Streamlit UI（1画面）
- `simulator.py`: 計算ロジック（既存ノートブック由来）
- `settings.py`: パスと初期設定（組成比候補など）
