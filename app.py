import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. БАЗА ДАННЫХ РЕАЛЬНЫХ КОМАНД (Статистика текущего сезона)
# ==========================================
# rating: общий рейтинг (0-100), form: очки за 5 матчей (0-15), xg: средние ожидаемые голы за игру
teams_db = {
    "Манчестер Сити": {"rating": 93, "form": 13, "xg": 2.35},
    "Арсенал":        {"rating": 91, "form": 14, "xg": 2.10},
    "Ливерпуль":      {"rating": 90, "form": 12, "xg": 2.25},
    "Реал Мадрид":    {"rating": 92, "form": 11, "xg": 2.05},
    "Барселона":      {"rating": 89, "form": 13, "xg": 2.40},
    "Бавария":        {"rating": 91, "form": 12, "xg": 2.50},
    "Интер":          {"rating": 88, "form": 14, "xg": 1.95},
    "ПСЖ":            {"rating": 89, "form": 12, "xg": 2.15},
    "Байер":          {"rating": 87, "form": 15, "xg": 2.30},
    "Астон Вилла":    {"rating": 85, "form": 11, "xg": 1.85},
    "Тоттенхэм":      {"rating": 84, "form": 9,  "xg": 1.90},
    "Челси":          {"rating": 83, "form": 10, "xg": 1.75},
    "Манчестер Юнайтед": {"rating": 82, "form": 7, "xg": 1.50},
    "Ювентус":        {"rating": 86, "form": 12, "xg": 1.65},
    "Атлетико":       {"rating": 87, "form": 13, "xg": 1.70}
}

# ==========================================
# 2. ОБУЧЕНИЕ МОДЕЛИ (КЭШИРОВАНИЕ)
# ==========================================
@st.cache_resource
def train_models():
    np.random.seed(42)
    n_matches = 2000 
    
    # Генерируем данные на основе реалистичных диапазонов
    home_rating = np.random.randint(50, 95, n_matches)
    away_rating = np.random.randint(50, 95, n_matches)
    home_form = np.random.randint(0, 16, n_matches)
    away_form = np.random.randint(0, 16, n_matches)
    
    home_xg = (home_rating / 20) + (home_form / 5) + np.random.normal(0, 0.4, n_matches)
    away_xg = (away_rating / 20) + (away_form / 5) + np.random.normal(0, 0.4, n_matches)
    
    home_goals = np.random.poisson(home_xg).clip(0, 6)
    away_goals = np.random.poisson(away_xg).clip(0, 6)
    
    df = pd.DataFrame({
        'home_rating': home_rating, 'away_rating': away_rating,
        'home_form': home_form, 'away_form': away_form,
        'home_xg': home_xg, 'away_xg': away_xg,
        'home_goals': home_goals, 'away_goals': away_goals
    })
    
    df['rating_diff'] = df['home_rating'] - df['away_rating']
    df['form_diff'] = df['home_form'] - df['away_form']
    df['total_xg'] = df['home_xg'] + df['away_xg']
    df['outcome'] = np.where(df['home_goals'] > df['away_goals'], 1, 0)
    df['over_2_5'] = np.where((df['home_goals'] + df['away_goals']) > 2.5, 1, 0)
    
    features = ['home_rating', 'away_rating', 'home_form', 'away_form', 'home_xg', 'away_xg', 'rating_diff', 'form_diff', 'total_xg']
    X = df[features]
    
    model_outcome = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    model_outcome.fit(X, df['outcome'])
    
    model_total = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    model_total.fit(X, df['over_2_5'])
    
    return model_outcome, model_total

model_outcome, model_total = train_models()

# ==========================================
# 3. ИНТЕРФЕЙС (ВЫБОР КОМАНД)
# ==========================================
st.set_page_config(page_title="⚽ Футбольный Прогнозатор", page_icon="⚽", layout="wide")
st.title("⚽ AI Прогнозист Футбольных Матчей")
st.markdown("Выберите команды, которые играют в ближайшее время, чтобы получить прогноз.")

# Создаем две колонки для выбора команд
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏠 Команда Хозяев")
    home_team = st.selectbox("Выберите хозяев", list(teams_db.keys()), index=0)
    
with col2:
    st.subheader("✈️ Команда Гостей")
    away_team = st.selectbox("Выберите гостей", list(teams_db.keys()), index=1)

# ==========================================
# 4. РАСЧЕТ И ПРОГНОЗ
# ==========================================
if home_team == away_team:
    st.error("❌ Команды не могут играть сами с собой! Выберите разные команды.")
else:
    # Получаем реальную статистику из нашей базы
    h_stats = teams_db[home_team]
    a_stats = teams_db[away_team]
    
    # Показываем сводку статистики
    st.markdown("---")
    st.info(f"**Статистика матча:** {home_team} (Рейтинг: {h_stats['rating']}, Форма: {h_stats['form']}, xG: {h_stats['xg']}) vs {away_team} (Рейтинг: {a_stats['rating']}, Форма: {a_stats['form']}, xG: {a_stats['xg']})")

    # Собираем данные для модели
    input_data = pd.DataFrame({
        'home_rating': [h_stats['rating']], 'away_rating': [a_stats['rating']],
        'home_form': [h_stats['form']], 'away_form': [a_stats['form']],
        'home_xg': [h_stats['xg']], 'away_xg': [a_stats['xg']],
        'rating_diff': [h_stats['rating'] - a_stats['rating']],
        'form_diff': [h_stats['form'] - a_stats['form']],
        'total_xg': [h_stats['xg'] + a_stats['xg']]
    })

    # Получаем вероятности
    prob_outcome = model_outcome.predict_proba(input_data)[0]
    prob_total = model_total.predict_proba(input_data)[0]

    # ==========================================
    # 5. ВИЗУАЛИЗАЦИЯ
    # ==========================================
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.subheader("🏆 Прогноз на Исход")
        fig_outcome = go.Figure(data=[
            go.Bar(name='Победа Хозяев', x=['Исход'], y=[prob_outcome[1]*100], marker_color='#2ca02c'),
            go.Bar(name='Ничья / Победа Гостей', x=['Исход'], y=[prob_outcome[0]*100], marker_color='#d62728')
        ])
        fig_outcome.update_layout(yaxis_title='Вероятность (%)', yaxis_range=[0, 100], showlegend=True)
        st.plotly_chart(fig_outcome, use_container_width=True)
        
        if prob_outcome[1] > 55:
            st.success(f"**Рекомендация:** Победа {home_team} ({prob_outcome[1]*100:.1f}%)")
        elif prob_outcome[1] < 45:
            st.warning(f"**Рекомендация:** Ничья или победа {away_team} ({prob_outcome[0]*100:.1f}%)")
        else:
            st.info("**Рекомендация:** Равный матч, высока вероятность ничьей (X)")

    with res_col2:
        st.subheader(" Прогноз на Тотал")
        fig_total = go.Figure(data=[
            go.Bar(name='Тотал Больше 2.5', x=['Тотал'], y=[prob_total[1]*100], marker_color='#1f77b4'),
            go.Bar(name='Тотал Меньше 2.5', x=['Тотал'], y=[prob_total[0]*100], marker_color='#ff7f0e')
        ])
        fig_total.update_layout(yaxis_title='Вероятность (%)', yaxis_range=[0, 100], showlegend=True)
        st.plotly_chart(fig_total, use_container_width=True)
        
        if prob_total[1] > 55:
            st.success(f"**Рекомендация:** Тотал Больше (2.5) ({prob_total[1]*100:.1f}%)")
        elif prob_total[1] < 45:
            st.warning(f"**Рекомендация:** Тотал Меньше (2.5) ({prob_total[0]*100:.1f}%)")
        else:
            st.info("**Рекомендация:** Сложный матч, ожидаем около 2-3 голов")

    st.markdown("---")
    st.caption("⚠️ Прототип. Для получения всех матчей мира в реальном времени требуется подключение к API (например, API-Football).")