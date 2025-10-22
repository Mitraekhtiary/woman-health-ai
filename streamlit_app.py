# streamlit_womens_health_app.py
# Smart Women's Health Analyzer
# Usage (Streamlit): streamlit run streamlit_womens_health_app.py
# Fallback usage (no streamlit): python streamlit_womens_health_app.py --file health_data.csv

"""
این نسخه طوری نوشته شده که در صورت عدم نصب streamlit (خطای ModuleNotFoundError)
به‌صورت fallback در محیط خط فرمان یا Jupyter اجرا شود و همان تحلیل‌ها را تولید کند.

قابلیت‌ها:
- خواندن CSV یا Excel
- تشخیص خودکار ستون هدف (مثلاً *score, balance, stress, anemia, risk*)
- پیش‌‌پردازش ساده و تبدیل categorical به عدد
- آموزش RandomForest و گزارش R² و MAE
- ذخیره‌سازی نمودارها به‌صورت HTML (Plotly) و CSV خروجی با پیش‌بینی
- تابع تعاملی ساده (CLI) برای پیش‌بینی یک رکورد جدید

نکته: اگر می‌خواهید نسخهٔ تحت‌وب (streamlit) را اجرا کنید، ابتدا streamlit را نصب کنید:
    pip install streamlit

"""

import sys
import os
import argparse
import math
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Try to import streamlit; if not available, fall back to CLI/notebook mode
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False

# -------------------------
# Core analysis functions
# -------------------------

def load_data(path):
    """Load CSV or Excel file into DataFrame."""
    if str(path).lower().endswith('.xlsx') or str(path).lower().endswith('.xls'):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    return df


def auto_detect_target(df):
    """Return a target column name detected automatically or None."""
    candidates = [c for c in df.columns if any(k in c.lower() for k in ['score','balance','stress','anemia','risk'])]
    return candidates[0] if candidates else None


def preprocess(df, target):
    """Simple preprocessing: drop all-empty rows, fill numeric NaNs with mean,
    convert non-numeric columns to categorical codes.
    Returns X, y and cleaned df.
    """
    df = df.dropna(how='all').copy()
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in data.")

    # fill target NaNs with mean
    y = df[target].copy()
    if y.isnull().any():
        y = y.fillna(y.mean())

    X = df.drop(columns=[target]).copy()

    # convert booleans to int
    for col in X.select_dtypes(include=['bool']).columns:
        X[col] = X[col].astype(int)

    # convert non-numeric to categorical codes
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype('category').cat.codes

    # drop columns that are all NaN or constant
    X = X.dropna(axis=1, how='all')
    nunique = X.nunique()
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        X = X.drop(columns=const_cols)

    return X, y, df


def train_and_report(X, y, test_size=0.2, n_estimators=120, random_state=42, output_prefix='output'):
    """Train RandomForestRegressor and produce metrics and visual outputs.
    Returns model and dict of metrics and file paths for saved artifacts.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))
    model = RandomForestRegressor(n_estimators=int(n_estimators), random_state=int(random_state))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Feature importances
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)

    # Save importance plot as HTML
    fig_imp = px.bar(importances, orientation='h', title='Feature Importances')
    imp_html = f"{output_prefix}_feature_importances.html"
    pio.write_html(fig_imp, file=imp_html, auto_open=False)

    # If number of features is small, save correlation heatmap
    heatmap_html = None
    if X.shape[1] <= 25:
        corr = pd.concat([X, y], axis=1).corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect='auto', title='Correlation heatmap')
        heatmap_html = f"{output_prefix}_correlation_heatmap.html"
        pio.write_html(fig_corr, file=heatmap_html, auto_open=False)

    # Approximate mean effect when setting to 75th percentile
    base = model.predict(X_test).mean()
    effects = {}
    for col in X.columns:
        X_tmp = X_test.copy()
        q = X_tmp[col].quantile(0.75)
        X_tmp[col] = q
        effects[col] = model.predict(X_tmp).mean() - base
    effects_ser = pd.Series(effects).sort_values()
    fig_eff = px.bar(effects_ser, orientation='h', title='Approx. mean effect when setting feature to 75th percentile')
    eff_html = f"{output_prefix}_effects.html"
    pio.write_html(fig_eff, file=eff_html, auto_open=False)

    artifacts = {
        'r2': r2,
        'mae': mae,
        'feature_importances': importances,
        'feature_importances_html': imp_html,
        'correlation_html': heatmap_html,
        'effects_html': eff_html,
        'model': model,
    }
    return artifacts


def predict_interactive(model, X_columns):
    """CLI interactive prompt to enter values for each feature and get prediction."""
    print('\nEnter values for the following features (press Enter to use mean):')
    vals = {}
    for col in X_columns:
        while True:
            try:
                s = input(f"  {col}: ")
                if s.strip() == '':
                    vals[col] = None
                    break
                else:
                    vals[col] = float(s)
                    break
            except Exception as e:
                print('ورودی نامعتبر، لطفاً مقدار عددی وارد کنید یا Enter بزنید.')
    # Build DataFrame
    df_in = pd.DataFrame([vals])
    # replace Nones with NaN
    df_in = df_in.astype('float')
    return df_in

# -------------------------
# CLI / Notebook fallback
# -------------------------

def run_cli(file_path, target=None, test_size=0.2, n_estimators=120, random_state=42, output_prefix='output'):
    print('Running Smart Women\'s Health Analyzer in CLI/fallback mode')
    print(f'Loading file: {file_path}')
    df = load_data(file_path)
    print('Columns found:', list(df.columns))

    if target is None:
        target = auto_detect_target(df)
        if target:
            print(f"Auto-detected target column: {target}")
        else:
            raise ValueError('No target column detected automatically. Please provide the --target argument.')
    else:
        print(f'Using target column: {target}')

    X, y, df_clean = preprocess(df, target)
    print(f'Features used: {list(X.columns)}')
    artifacts = train_and_report(X, y, test_size=test_size, n_estimators=n_estimators, random_state=random_state, output_prefix=output_prefix)

    print('\nModel performance:')
    print(f"  R2 = {artifacts['r2']:.4f}")
    print(f"  MAE = {artifacts['mae']:.4f}")
    print('\nSaved artifacts:')
    print('  - Feature importances HTML:', artifacts['feature_importances_html'])
    if artifacts['correlation_html']:
        print('  - Correlation heatmap HTML:', artifacts['correlation_html'])
    print('  - Effects HTML:', artifacts['effects_html'])

    # Save predictions for full dataset
    preds = artifacts['model'].predict(X)
    out_df = df_clean.copy()
    out_df['predicted_' + str(target)] = preds
    out_csv = f"{output_prefix}_predictions.csv"
    out_df.to_csv(out_csv, index=False)
    print('  - Predictions CSV:', out_csv)

    # Ask user if they want to do interactive prediction
    do_interactive = input('\nDo interactive prediction? (y/N): ').strip().lower() == 'y'
    if do_interactive:
        df_in = predict_interactive(artifacts['model'], X.columns)
        # fill missing with column means
        for c in X.columns:
            if c not in df_in.columns or pd.isna(df_in.at[0,c]):
                df_in.at[0,c] = X[c].mean()
        pred = artifacts['model'].predict(df_in[X.columns])[0]
        print(f"\nPrediction for input: {pred:.2f}")

    print('\nDone.')

# -------------------------
# Streamlit UI (if available)
# -------------------------

def run_streamlit_app():
    # This function contains the same logic as before but only runs when streamlit is available
    st.set_page_config(page_title="Smart Women's Health Analyzer", layout='wide')
    st.title("🌺 Smart Women's Health Analyzer")
    st.write("آپلود فایل CSV/XLSX شامل داده‌های سبک زندگی و یک ستون هدف (مثلاً Hormonal_balance_score یا Stress_level)")

    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv','xlsx'])

    if uploaded_file is not None:
        try:
            if str(uploaded_file.name).endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"خطا در خواندن فایل: {e}")
            st.stop()

        st.subheader("📌 پیش‌نمایش داده")
        st.dataframe(df.head())
        st.markdown(f"**تعداد ردیف:** {len(df)}  —  **ستون‌ها:** {', '.join(df.columns)}")

        df = df.dropna(how='all')
        candidates = [c for c in df.columns if any(k in c.lower() for k in ['score','balance','stress','anemia','risk'])]

        with st.sidebar:
            st.header("تنظیمات مدل")
            if candidates:
                target = st.selectbox('انتخاب ستون هدف (در صورت یافتن خودکار)', ['--auto--'] + candidates)
                if target == '--auto--':
                    target = candidates[0]
            else:
                target = st.text_input('نام ستون هدف (مثلاً Hormonal_balance_score)')

            test_size = st.slider('نسبت تست (test size)', min_value=0.1, max_value=0.4, value=0.2)
            n_estimators = st.slider('تعداد درختان RandomForest', min_value=10, max_value=300, value=120, step=10)
            random_state = st.number_input('Random state (int)', value=42, step=1)
            st.markdown('---')
            st.markdown('برای ستون‌های غیرعددی، آن‌ها را در پیش‌پردازش به عدد تبدیل کنید (one-hot یا label).')

        if target not in df.columns:
            st.error('ستون هدف معتبر نیست. لطفاً ستون موجود را انتخاب کنید.')
            st.stop()

        X = df.drop(columns=[target]).copy()
        y = df[target].copy()

        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].astype('category').cat.codes

        X = X.dropna(axis=1, how='all')
        y = y.fillna(y.mean())

        st.subheader('📈 مصورسازی و بررسی همبستگی')
        if X.shape[1] <= 8:
            corr = pd.concat([X, y], axis=1).corr()
            fig_corr = px.imshow(corr, text_auto=True, aspect='auto', title='Heatmap همبستگی')
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info('ستون‌های بسیار زیاد برای نمایش heatmap — از جدول اهمیت ویژگی‌ها استفاده کنید.')

        col1, col2 = st.columns([1,1])
        with col1:
            st.subheader('توزیع هدف')
            fig = px.histogram(y, nbins=30, title=f'Distribution of {target}')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader('پیش‌نمایش ویژگی‌های عددی')
            st.write(X.describe().T)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))
        model = RandomForestRegressor(n_estimators=int(n_estimators), random_state=int(random_state))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        st.subheader('🔎 عملکرد مدل')
        st.metric('R² on test', f"{r2:.3f}")
        st.metric('MAE on test', f"{mae:.3f}")

        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
        st.subheader('💡 اهمیت ویژگی‌ها')
        fig_imp = px.bar(importances, orientation='h', title='Feature Importances')
        st.plotly_chart(fig_imp, use_container_width=True)

        st.subheader('🔍 تأثیر میانگین هر ویژگی (خلاصه)')
        effects = {}
        base = model.predict(X_test).mean()
        for col in X.columns:
            X_tmp = X_test.copy()
            q = X_tmp[col].quantile(0.75)
            X_tmp[col] = q
            effects[col] = model.predict(X_tmp).mean() - base
        effects_ser = pd.Series(effects).sort_values()
        fig_eff = px.bar(effects_ser, orientation='h', title='Approx. mean effect when setting feature to 75th percentile')
        st.plotly_chart(fig_eff, use_container_width=True)

        st.subheader('🧪 پیش‌بینی تعاملی برای رکورد جدید')
        with st.form(key='input_form'):
            input_data = {}
            for col in X.columns:
                if pd.api.types.is_integer_dtype(X[col]) or pd.api.types.is_float_dtype(X[col]):
                    min_v = float(X[col].min())
                    max_v = float(X[col].max())
                    mean_v = float(X[col].mean())
                    input_data[col] = st.number_input(col, value=mean_v, min_value=min_v, max_value=max_v)
                else:
                    input_data[col] = st.number_input(col, value=0)
            submit = st.form_submit_button('پیش‌بینی کن')

        if submit:
            x_new = pd.DataFrame([input_data])[X.columns]
            pred = model.predict(x_new)[0]
            st.success(f'🔮 پیش‌بینی {target}: {pred:.2f}')

            top_feats = importances.tail(3).index.tolist()[::-1]
            st.markdown('**پیشنهادهای ابتدایی (قواعد ساده):**')
            for f in top_feats:
                val = x_new[f].iloc[0]
                st.write(f"- {f}: مقدار فعلی = {val}")
                if 'sleep' in f.lower() and val < X[f].mean():
                    st.write('  - توصیه: افزایش خواب به نزدیک میانگین جمع‌آوری‌شده.')
                if 'caffe' in f.lower() and val > X[f].mean():
                    st.write('  - توصیه: کاهش مصرف کافئین.')

        if st.button('تولید پیش‌بینی برای کل دیتاست و دانلود CSV'):
            df_out = df.copy()
            df_out['predicted_'+str(target)] = model.predict(X)
            st.download_button('دانلود CSV', data=df_out.to_csv(index=False).encode('utf-8'), file_name='predictions.csv')

        st.markdown('---')
        st.write('📌 نکات اخلاقی: داده‌های حساس را محافظت کنید. مدل نیاز به اعتبارسنجی کلینیکی دارد. این اپ توصیه پزشکی نیست.')
    else:
        st.info('لطفاً یک فایل داده آپلود کنید تا آنالیز آغاز شود.')

# -------------------------
# Entry point
# -------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Smart Women's Health Analyzer - fallback/streamlit app")
    parser.add_argument('--file', '-f', type=str, help='Path to CSV or Excel file')
    parser.add_argument('--target', '-t', type=str, default=None, help='Name of target column (optional)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test size fraction')
    parser.add_argument('--n-estimators', type=int, default=120, help='Number of trees for RandomForest')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed')
    parser.add_argument('--output-prefix', type=str, default='output', help='Prefix for saved artifacts')
    args = parser.parse_args()

    if STREAMLIT_AVAILABLE:
        # If streamlit is available, run the Streamlit app
        run_streamlit_app()
    else:
        # CLI mode
        if not args.file:
            print('Streamlit is not installed in this environment. Running in CLI/fallback mode.')
            print('Please provide a data file with --file data.csv or install streamlit with `pip install streamlit` to use the web UI.')
            sys.exit(1)
        run_cli(args.file, target=args.target, test_size=args.test_size, n_estimators=args.n_estimators, random_state=args.random_state, output_prefix=args.output_prefix)
