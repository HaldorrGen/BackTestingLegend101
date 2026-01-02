import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from data_feed import load_data
import numpy as np
import math
import os

def _run_single_backtest(ticker, start_date, end_date, ws, wl, watr, 
                         capital=100000, 
                         slope_lookback=1,  # период для проверки роста SMA
                         show_sma=True, 
                         show_equity=True, 
                         show_signals=True, 
                         show_atr=False, 
                         show_buy_and_hold=True,
                         show_plot=True,
                         save_metrics=True,
                         warm_up_days=0,
                         cash_rate=0.03,  # годовая доходность T-bills (например, 3%)
                         commission=0.0):  # комиссия за маркет ордеры (0.1% = 0.001)
    """
    Запускает бэктест стратегии на заданных параметрах
    
    Args:
        ticker: тикер инструмента
        start_date: начальная дата
        end_date: конечная дата
        ws: окно короткой SMA
        wl: окно длинной SMA
        watr: окно ATR
        capital: начальный капитал
        show_sma: показывать скользящие средние
        show_equity: показывать кривую капитала
        show_signals: показывать сигналы входа
        show_atr: показывать ATR
        show_buy_and_hold: показывать buy and hold
        show_plot: показывать график
        save_metrics: сохранять метрики в файл
        warm_up_days: количество торговых дней для прогрева индикаторов
        
    Returns:
        dict: словарь с метриками стратегии
    """
    
    data = load_data(ticker, start_date=start_date, end_date=end_date, ws=ws, wl=wl, watr=watr, warm_up_days=warm_up_days)
    data.index = pd.to_datetime(data.index)

    # Разделяем два разных концепта:
    lookahead_shift = 1  # сдвиг для предотвращения lookahead bias (используем сигнал следующего дня)
    # slope_lookback: передается как параметр функции

    # Условия входа в позицию
    cond_price_above_long = data["Close"] > data["SMA_long"]
    cond_short_above_long = data["SMA_short"] > data["SMA_long"]
    cond_short_up = data["SMA_short"] > data["SMA_short"].shift(slope_lookback)  # SMA растет за N дней
    trend_long = cond_price_above_long & cond_short_above_long & cond_short_up
    
    # Сигнал: 1 = long, 0 = в кэше
    data["signal"] = np.where(trend_long, 1, 0)

    # Рассчитываем доходность стратегии с правильным компаундингом
    # ТОРГОВЛЯ ВЕДЕТСЯ С ПЕРВОГО ДНЯ (включая warm-up)
    # lookahead_shift: применяем сигнал только на следующий день (предотвращение lookahead bias)
    
    # Доходность актива (только когда в рынке)
    market_return = data["Close"].pct_change() * data["signal"].shift(lookahead_shift)
    
    # Комиссия за маркет ордеры: когда происходит переход в позицию или выход из позиции
    # Выход (1->0): платим комиссию при закрытии позиции
    # Вход (0->1): платим комиссию при открытии позиции
    signal_shifted = data["signal"].shift(lookahead_shift)
    signal_changes = signal_shifted.diff()
    
    # Комиссия применяется в день сделки (вход или выход)
    commission_cost = 0.0
    if commission > 0:
        # При входе в позицию (0 -> 1): платим комиссию
        # При выходе из позиции (1 -> 0): платим комиссию
        commission_cost = commission * (signal_changes.abs() > 0.5)
    
    # Доходность cash (T-bills) когда вне рынка: используем правильное компаундирование
    # daily_rf = (1 + annual_rate)^(1/252) - 1
    daily_cash_rate = (1 + cash_rate) ** (1/252) - 1
    cash_return = daily_cash_rate * (1 - signal_shifted)
    
    # Итоговая доходность = рынок + кэш - комиссия
    data["daily_return"] = market_return + cash_return - commission_cost
    data["equity_curve"] = capital * (1 + data["daily_return"]).cumprod()

    # --- Buy & Hold equity для сравнения ---
    # ВАЖНО: берем первый Close из analysis_period, а не из всех данных
    # Сначала найдем первый день анализа
    first_analysis_close = data[data["is_analysis_period"]].iloc[0]["Close"] if data["is_analysis_period"].any() else data["Close"].iloc[0]
    data["buy_hold_equity"] = capital * (data["Close"] / first_analysis_close)

    # --- МЕТРИКИ СЧИТАЕМ ТОЛЬКО НА ПЕРИОДЕ АНАЛИЗА (после warm-up) ---
    analysis_data = data[data["is_analysis_period"]].copy()
    
    if len(analysis_data) == 0:
        raise ValueError("Нет данных в периоде анализа после warm-up")

    # начальная и конечная дата (РЕАЛЬНОГО периода анализа)
    start_date_actual = analysis_data.index.min()
    end_date_actual = analysis_data.index.max()
    
    # Начальный капитал для периода анализа = equity на конец warm-up
    # (или исходный capital если warm-up не было)
    if warm_up_days > 0:
        warm_up_data = data[~data["is_analysis_period"]]
        if len(warm_up_data) > 0:
            capital_at_analysis_start = data.loc[data["is_analysis_period"]].iloc[0]["equity_curve"]
        else:
            capital_at_analysis_start = capital
    else:
        capital_at_analysis_start = capital

    # Рассчитываем количество дней и лет
    if isinstance(start_date_actual, pd.Timestamp):
        days = (end_date_actual - start_date_actual).days
    else:
        # Если индекс не datetime, используем количество строк как приблизительную оценку
        days = len(analysis_data)
        
    years = days / 365.25 if days > 0 else 1

    # финальные значения equity (на конец периода анализа)
    final_equity_strategy = analysis_data["equity_curve"].iloc[-1]
    final_equity_bh = analysis_data["buy_hold_equity"].iloc[-1]

    # total return (от начала анализа, не от начального capital)
    total_ret_strategy = (final_equity_strategy / capital_at_analysis_start - 1) if capital_at_analysis_start > 0 else 0
    # Для buy-and-hold пересчитываем относительно начала периода анализа
    bh_start = analysis_data["buy_hold_equity"].iloc[0]
    total_ret_bh = (final_equity_bh / bh_start - 1) if bh_start > 0 else 0

    # CAGR (от начала анализа)
    cagr_strategy = ((final_equity_strategy / capital_at_analysis_start) ** (1 / years) - 1) if years > 0 and capital_at_analysis_start > 0 else np.nan
    cagr_bh = ((final_equity_bh / bh_start) ** (1 / years) - 1) if years > 0 and bh_start > 0 else np.nan

    # Max Drawdown стратегии (только на периоде анализа)
    equity = analysis_data["equity_curve"].dropna()
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1
    max_dd = drawdown.min() if not drawdown.empty else np.nan

    # Sharpe ratio (дневной, annualized) - только на периоде анализа
    daily_ret = analysis_data["daily_return"].dropna()
    if len(daily_ret) > 1 and daily_ret.std() > 1e-12:
        sharpe = (daily_ret.mean() / daily_ret.std()) * math.sqrt(252)
    else:
        sharpe = np.nan  # Защита от нулевой волатильности
    
    # --- НОВЫЕ МЕТРИКИ ---
    
    # % days in market (экспозиция)
    # Используем lookahead_shift, потому что реально торгуем со сдвигом
    days_in_market = (analysis_data["signal"].shift(lookahead_shift) == 1).sum()
    total_days = len(analysis_data)
    market_exposure = (days_in_market / total_days * 100) if total_days > 0 else 0
    
    # Количество сделок (входов)
    # Сделка = переход из 0 в 1
    signal_changes = analysis_data["signal"].diff()
    num_trades = (signal_changes == 1).sum()

    # Подготовим текстовый блок
    metrics_text = (
        f"Ticker: {ticker}\n"
        f"Период анализа: {start_date_actual.date()} — {end_date_actual.date()}\n"
        f"Лет: {years:.2f} | Торговых дней: {total_days}\n"
        f"Параметры: SMA_short={ws}, SMA_long={wl}, ATR={watr}, slope_lookback={slope_lookback}\n"
    )
    
    if warm_up_days > 0:
        metrics_text += f"Warm-up: {warm_up_days} дней (торговля велась с warm-up)\n"
        metrics_text += f"Начальный капитал для анализа: {capital_at_analysis_start:,.0f} $\n"
    
    metrics_text += (
        f"\nСтратегия (на периоде анализа):\n"
        f"  Final equity: {final_equity_strategy:,.0f} $\n"
        f"  Total return: {total_ret_strategy*100:,.1f} %\n"
        f"  CAGR:        {cagr_strategy*100:,.2f} %\n"
        f"  Max DD:      {max_dd*100:,.2f} %\n"
        f"  Sharpe:      {sharpe:.2f}\n"
        f"  Экспозиция:  {market_exposure:.1f}% ({days_in_market} дней)\n"
        f"  Сделок:      {num_trades}\n\n"
        f"Buy & Hold (на периоде анализа):\n"
        f"  Final equity: {final_equity_bh:,.0f} $\n"
        f"  Total return: {total_ret_bh*100:,.1f} %\n"
        f"  CAGR:        {cagr_bh*100:,.2f} %"
    )
    

    # Создаем словарь с метриками для возврата
    metrics = {
        "ticker": ticker,
        "start_date": start_date_actual,
        "end_date": end_date_actual,
        "years": years,
        "ws": ws,
        "wl": wl,
        "watr": watr,
        "final_equity_strategy": final_equity_strategy,
        "final_equity_bh": final_equity_bh,
        "total_return_strategy": total_ret_strategy,
        "total_return_bh": total_ret_bh,
        "cagr_strategy": cagr_strategy,
        "cagr_bh": cagr_bh,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "market_exposure": market_exposure,
        "num_trades": num_trades,
        "days_in_market": days_in_market,
        "total_days": total_days,
        "data": data  # возвращаем данные для дальнейшего анализа
    }

    print("\n" + "="*60)
    print(metrics_text)
    print("="*60 + "\n")

    # --- ГРАФИКИ ---
    if show_plot:
        # Определяем количество субплотов
        num_plots = 1 + (1 if show_atr else 0)
        fig, axes = plt.subplots(num_plots, 1, figsize=(14, 8 * num_plots / 1.5), sharex=True)

        # Если только один график, axes не будет массивом
        if num_plots == 1:
            axes = [axes]

        ax_price = axes[0]

        # 1) Цена
        ax_price.plot(data.index, data["Close"], label="Close Price", color="black", linewidth=1.5)

        # Скользящие средние
        if show_sma:
            ax_price.plot(data.index, data["SMA_short"], label=f"SMA {ws}", color="orange", linewidth=1, alpha=0.7)
            ax_price.plot(data.index, data["SMA_long"], label=f"SMA {wl}", color="red", linewidth=1, alpha=0.7)

        # Сигналы входа и выхода - вертикальные линии
        if show_signals:
            # Находим переходы
            signal_changes = data["signal"].diff()
            
            # Вход в позицию (переход из 0 в 1) - зеленые линии
            long_entries = data.index[signal_changes == 1]
            for entry_date in long_entries:
                ax_price.axvline(x=entry_date, color='green', linewidth=0.8, alpha=0.6, linestyle='-')
            
            # Выход из позиции (переход из 1 в 0) - красные линии
            long_exits = data.index[signal_changes == -1]
            for exit_date in long_exits:
                ax_price.axvline(x=exit_date, color='red', linewidth=0.8, alpha=0.6, linestyle='-')
            
            # Добавляем легенду
            if len(long_entries) > 0:
                ax_price.plot([], [], color='green', linewidth=0.8, alpha=0.6, label='Entry')
            if len(long_exits) > 0:
                ax_price.plot([], [], color='red', linewidth=0.8, alpha=0.6, label='Exit')

        # Заголовок и оформление
        title = f"{ticker} Price (SMA {ws}/{wl})"
        if show_equity:
            title += " with Equity Curve"
        ax_price.set_title(title)
        if num_plots == 1:
            ax_price.set_xlabel("Date")
        ax_price.set_ylabel("Price ($)", color="black")
        ax_price.tick_params(axis='y', labelcolor="black")
        ax_price.grid(True, alpha=0.3)
        ax_price.legend(loc='upper left')

        # 2) Кривая капитала на второй оси Y
        if show_equity:
            ax_equity = ax_price.twinx()
            ax_equity.plot(data.index, data["equity_curve"], label="Equity Curve", color="blue", alpha=0.7, linewidth=2)
            
            # 2.1) Buy and Hold стратегия (для сравнения)
            if show_buy_and_hold:
                ax_equity.plot(data.index, data["buy_hold_equity"], 
                             label="Buy and Hold", color="gray", linestyle="--", alpha=0.7, linewidth=2)
            
            ax_equity.set_ylabel("Equity ($)", color="blue")
            ax_equity.tick_params(axis='y', labelcolor="blue")
            ax_equity.legend(loc='upper right')

        # 3) ATR (опционально на отдельной панели)
        if show_atr:
            ax_atr = axes[1]
            ax_atr.plot(data.index, data["ATR"], label="ATR", color="purple", linewidth=1.5)
            ax_atr.set_title(f"{ticker} ATR")
            ax_atr.set_xlabel("Date")
            ax_atr.set_ylabel("ATR", color="purple")
            ax_atr.tick_params(axis='y', labelcolor="purple")
            ax_atr.grid(True, alpha=0.3)
            ax_atr.legend(loc='upper left')

        # --- ПАНЕЛЬ ТЕКСТА СПРАВА ---
        # добавляем отдельную ось справа от графиков
        text_ax = fig.add_axes([0.78, 0.15, 0.2, 0.7])  # [left, bottom, width, height]
        text_ax.axis("off")
        text_ax.text(0, 1, metrics_text, va="top", fontsize=10, family='monospace')

        plt.tight_layout(rect=[0, 0, 0.76, 1])  # оставляем место справа под текстовую панель
        plt.show()
    
    return metrics


def run_portfolio_backtest(tickers, start_date, end_date, ws, wl, watr,
                          capital=100000,
                          slope_lookback=1,  # период для проверки роста SMA
                          allocation='risk_parity',  # 'equal', 'risk_parity' или словарь {ticker: weight}
                          volatility_target=0.10,  # целевая волатильность портфеля (например, 0.10 = 10%)
                          volatility_lookback=252,  # окно для расчета волатильности (дней)
                          show_individual_plots=False,
                          show_portfolio_plot=True,
                          save_metrics=True,
                          warm_up_days=0,
                          cash_rate=0.03,  # годовая доходность T-bills
                          leverage_cap=2.0,  # максимальный коэффициент левериджа для vol targeting
                          normalize_by='usd',  # 'usd' или 'gold' - валюта отображения графиков
                          commission=0.0):  # комиссия за маркет ордеры (0.1% = 0.001)
    
    if isinstance(tickers, str):
        tickers = [tickers]
    
    # --- РАСЧЕТ ВОЛАТИЛЬНОСТЕЙ ДЛЯ RISK PARITY ---
    if allocation == 'risk_parity':
        print(f"\n{'='*80}")
        print(f"РАСЧЕТ ВЕСОВ ПО RISK PARITY (STATIC)")
        print(f"{'='*80}\n")
        
        # Загружаем данные для расчета волатильности
        volatilities = {}
        usable_tickers = []
        
        for ticker in tickers:
            try:
                # Загружаем данные с warm_up для индикаторов
                temp_data = load_data(ticker, start_date, end_date, ws=ws, wl=wl, watr=watr, 
                                     warm_up_days=warm_up_days)
                
                # ВОЛАТИЛЬНОСТЬ СЧИТАЕМ ЗА volatility_lookback дней
                # Приоритет: берем данные ДО analysis period (если есть warm-up)
                # Иначе: берем начало analysis period
                
                pre_analysis_data = temp_data[~temp_data['is_analysis_period']]
                analysis_data_for_vol = temp_data[temp_data['is_analysis_period']]
                
                if len(pre_analysis_data) >= volatility_lookback:
                    # Есть достаточно данных ДО анализа - берем их
                    vol_window = pre_analysis_data.iloc[-volatility_lookback:]
                    daily_returns = vol_window['Close'].pct_change().dropna()
                    vol_source = f"warm-up ({len(daily_returns)} дней)"
                elif len(pre_analysis_data) > 0:
                    # Есть немного warm-up данных - используем их
                    daily_returns = pre_analysis_data['Close'].pct_change().dropna()
                    vol_source = f"warm-up ({len(daily_returns)} дней, < {volatility_lookback})"
                else:
                    # Нет warm-up - берем начало периода анализа
                    if len(analysis_data_for_vol) >= volatility_lookback:
                        vol_window = analysis_data_for_vol.iloc[:volatility_lookback]
                    else:
                        vol_window = analysis_data_for_vol
                    daily_returns = vol_window['Close'].pct_change().dropna()
                    vol_source = f"начало анализа ({len(daily_returns)} дней)"
                
                # Аннуализированная волатильность
                if len(daily_returns) > 1:
                    vol = daily_returns.std() * math.sqrt(252)
                    volatilities[ticker] = vol
                    usable_tickers.append(ticker)
                    print(f"{ticker}: волатильность = {vol*100:.2f}% (источник: {vol_source})")
                else:
                    raise ValueError(f"Недостаточно данных для расчета волатильности {ticker}")
                
            except Exception as e:
                print(f"⚠️ Пропускаем {ticker}: {e}")
                continue
        
        if not usable_tickers:
            raise ValueError("Нет тикеров с валидными данными для расчета волатильности")
        tickers = usable_tickers
        
        # Расчитываем веса: пропорционально 1/волатильности
        inverse_vols = {ticker: 1.0 / vol for ticker, vol in volatilities.items()}
        total_inverse_vol = sum(inverse_vols.values())
        
        weights = {ticker: inv_vol / total_inverse_vol for ticker, inv_vol in inverse_vols.items()}
        tickers = list(weights.keys())
        
        print(f"\nВеса по risk parity (static):")
        for ticker, weight in weights.items():
            risk_contribution = weight * volatilities[ticker]
            print(f"  {ticker}: {weight*100:.2f}% (вол: {volatilities[ticker]*100:.2f}%, risk contrib: {risk_contribution*100:.2f}%)")
        
        # Расчет ожидаемой волатильности портфеля (упрощенная формула при нулевой корреляции)
        # Правильно: sqrt(sum((w_i * vol_i)^2)) при нулевой корреляции
        # Старая формула sum(w_i * vol_i) была upper bound, не волатильностью
        expected_portfolio_vol = math.sqrt(sum([(weights[t] * volatilities[t])**2 for t in tickers]))
        print(f"\nОжидаемая волатильность портфеля: {expected_portfolio_vol*100:.2f}% (при нулевой корреляции)")
        print(f"   Примечание: реальная вола будет зависеть от корреляций между активами")
        
        if volatility_target:
            print(f"Целевая волатильность: {volatility_target*100:.2f}%")
            print(f"⚠️  Примечание: Dynamic vol targeting пока не реализован (static risk parity only)")
        
        print(f"{'='*80}\n")
    
    # Определяем распределение капитала
    elif allocation == 'equal':
        weights = {ticker: 1.0 / len(tickers) for ticker in tickers}
    else:
        weights = allocation
        # Нормализуем веса
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
    
    print(f"\n{'='*80}")
    print(f"ПОРТФЕЛЬНЫЙ БЭКТЕСТ: {', '.join(tickers)}")
    print(f"Период: {start_date} - {end_date}")
    print(f"Распределение капитала: {', '.join([f'{t}: {w*100:.1f}%' for t, w in weights.items()])}")
    print(f"{'='*80}\n")
    
    # Запускаем бэктест для каждого тикера
    ticker_results = {}
    ticker_capital = {ticker: capital * weights[ticker] for ticker in tickers}
    
    for ticker in tickers:
        print(f"\n--- Анализ {ticker} (капитал: ${ticker_capital[ticker]:,.0f}) ---")
        try:
            result = _run_single_backtest(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                ws=ws,
                wl=wl,
                watr=watr,
                capital=ticker_capital[ticker],
                slope_lookback=slope_lookback,
                show_plot=show_individual_plots,
                save_metrics=False,
                warm_up_days=warm_up_days,
                cash_rate=cash_rate,
                commission=commission
            )
            ticker_results[ticker] = result
        except Exception as e:
            print(f"❌ Ошибка при обработке {ticker}: {e}")
            continue
    
    if not ticker_results:
        raise ValueError("Не удалось обработать ни один тикер")
    
    # --- РАСЧЕТ ПОРТФЕЛЬНЫХ МЕТРИК ---
    
    # Собираем все даты (пересечение дат всех тикеров)
    all_dates = None
    for ticker, result in ticker_results.items():
        ticker_dates = result['data'][result['data']['is_analysis_period']].index
        if all_dates is None:
            all_dates = set(ticker_dates)
        else:
            all_dates = all_dates.intersection(set(ticker_dates))
    
    all_dates = sorted(list(all_dates))
    
    if len(all_dates) == 0:
        raise ValueError("Нет общих дат для всех тикеров")
    
    # Создаем общую equity curve портфеля
    portfolio_equity = pd.Series(0.0, index=all_dates)
    portfolio_bh_equity = pd.Series(0.0, index=all_dates)
    
    for ticker, result in ticker_results.items():
        data = result['data']
        analysis_data = data[data['is_analysis_period']]
        
        # Берем только общие даты
        ticker_equity = analysis_data.loc[all_dates, 'equity_curve']
        ticker_bh = analysis_data.loc[all_dates, 'buy_hold_equity']
        
        portfolio_equity += ticker_equity
        portfolio_bh_equity += ticker_bh
    
    # Рассчитываем метрики портфеля
    start_date_actual = pd.to_datetime(all_dates[0])
    end_date_actual = pd.to_datetime(all_dates[-1])
    days = (end_date_actual - start_date_actual).days
    years = days / 365.25 if days > 0 else 1
    
    # Total returns
    total_ret_portfolio = (portfolio_equity.iloc[-1] / capital - 1) if capital > 0 else 0
    total_ret_bh = (portfolio_bh_equity.iloc[-1] / capital - 1) if capital > 0 else 0
    
    # CAGR
    cagr_portfolio = ((portfolio_equity.iloc[-1] / capital) ** (1 / years) - 1) if years > 0 and capital > 0 else np.nan
    cagr_bh = ((portfolio_bh_equity.iloc[-1] / capital) ** (1 / years) - 1) if years > 0 and capital > 0 else np.nan
    
    # Max Drawdown
    roll_max = portfolio_equity.cummax()
    drawdown = portfolio_equity / roll_max - 1
    max_dd = drawdown.min()
    
    # Sharpe
    daily_returns = portfolio_equity.pct_change().dropna()
    if len(daily_returns) > 1 and daily_returns.std() > 1e-12:
        sharpe = (daily_returns.mean() / daily_returns.std()) * math.sqrt(252)
    else:
        sharpe = np.nan  # Защита от нулевой волатильности
    
    # Exposure и trades - средние по тикерам (нужны ДО vol targeting для warning)
    avg_exposure = np.mean([r['market_exposure'] for r in ticker_results.values()])
    total_trades = sum([r['num_trades'] for r in ticker_results.values()])
    
    # --- DYNAMIC PORTFOLIO-LEVEL VOLATILITY TARGETING ---
    vol_target_warning = ""
    vol_targeting_window = 63  # окно для rolling volatility (примерно квартал)
    
    if volatility_target and len(daily_returns) > vol_targeting_window:
        # Рассчитываем rolling realized volatility (только на прошлых данных)
        rolling_vol = daily_returns.rolling(window=vol_targeting_window).std() * math.sqrt(252)
        
        # Рассчитываем динамический leverage coefficient k(t) для каждого дня
        # k(t) = clip(target_vol / realized_vol(t), 0, leverage_cap)
        # Применяется в ОБЕ СТОРОНЫ: снижает риск при высокой воле, повышает при низкой
        leverage_series = (volatility_target / rolling_vol).clip(0, leverage_cap)
        
        # Заполняем NaN в начале периода (где недостаточно данных для rolling)
        leverage_series = leverage_series.fillna(1.0)
        
        # Применяем k(t-1) к доходности дня t (избегаем lookahead bias)
        leverage_lagged = leverage_series.shift(1).fillna(1.0)
        daily_returns_targeted = daily_returns * leverage_lagged
        
        # Пересчитываем портфельную equity с динамическим таргетингом
        portfolio_equity_targeted = capital * (1 + daily_returns_targeted).cumprod()
        
        # Обновляем метрики
        portfolio_equity = portfolio_equity_targeted
        total_ret_portfolio = (portfolio_equity.iloc[-1] / capital - 1) if capital > 0 else 0
        cagr_portfolio = ((portfolio_equity.iloc[-1] / capital) ** (1 / years) - 1) if years > 0 and capital > 0 else np.nan
        
        # Max Drawdown
        roll_max = portfolio_equity.cummax()
        drawdown = portfolio_equity / roll_max - 1
        max_dd = drawdown.min()
        
        # Sharpe с targeted returns
        if len(daily_returns_targeted) > 1 and daily_returns_targeted.std() > 1e-12:
            sharpe = (daily_returns_targeted.mean() / daily_returns_targeted.std()) * math.sqrt(252)
        else:
            sharpe = np.nan
        
        # Realized vol после таргетинга
        realized_portfolio_vol = daily_returns_targeted.std() * math.sqrt(252) if len(daily_returns_targeted) > 1 else np.nan
        
        # Средний leverage за период
        leverage_applied = leverage_lagged.mean()
        
        # Предупреждение если низкая экспозиция ограничивает достижение vol target
        if avg_exposure < 30 and leverage_applied >= (leverage_cap * 0.9):
            vol_target_warning = f"\n⚠️ ВНИМАНИЕ: Низкая экспозиция ({avg_exposure:.1f}%) ограничивает достижение целевой волатильности.\n   Средний leverage: {leverage_applied:.2f}x (близко к cap {leverage_cap:.2f}x)\n   Рассмотрите добавление defensive assets (bonds/cash ETF) для улучшения vol targeting.\n"
    else:
        # Если недостаточно данных для vol targeting или target не задан
        leverage_applied = 1.0
        realized_portfolio_vol = daily_returns.std() * math.sqrt(252) if len(daily_returns) > 1 else np.nan
    
    # Формируем отчет
    metrics_text = (
        f"ПОРТФЕЛЬ: {', '.join(tickers)}\n"
        f"Период: {start_date_actual.date()} — {end_date_actual.date()}\n"
        f"Лет: {years:.2f} | Торговых дней: {len(all_dates)}\n"
        f"Параметры: SMA_short={ws}, SMA_long={wl}, ATR={watr}, slope_lookback={slope_lookback}\n"
        f"Cash rate (T-bills): {cash_rate*100:.2f}% годовых\n"
    )
    
    if vol_target_warning:
        metrics_text += vol_target_warning
    
    if warm_up_days > 0:
        metrics_text += f"Warm-up: {warm_up_days} дней\n"
    
    metrics_text += (
        f"\nПортфель (общие метрики):\n"
        f"  Начальный капитал: {capital:,.0f} $\n"
        f"  Final equity:      {portfolio_equity.iloc[-1]:,.0f} $\n"
        f"  Total return:      {total_ret_portfolio*100:,.1f} %\n"
        f"  CAGR:              {cagr_portfolio*100:,.2f} %\n"
        f"  Max DD:            {max_dd*100:,.2f} %\n"
        f"  Sharpe:            {sharpe:.2f}\n"
        f"  Realized Vol:      {realized_portfolio_vol*100:.2f}%\n"
        f"  Target Vol:        {volatility_target*100:.2f}%\n"
        f"  Leverage applied:  {leverage_applied:.2f}x\n"
        f"  Ср. экспозиция:    {avg_exposure:.1f}%\n"
        f"  Всего сделок:      {total_trades}\n\n"
        f"Buy & Hold портфеля:\n"
        f"  Final equity:      {portfolio_bh_equity.iloc[-1]:,.0f} $\n"
        f"  Total return:      {total_ret_bh*100:,.1f} %\n"
        f"  CAGR:              {cagr_bh*100:,.2f} %\n\n"
    )
    
    # Добавляем информацию по отдельным тикерам
    metrics_text += "="*60 + "\n"
    metrics_text += "ДЕТАЛИЗАЦИЯ ПО ТИКЕРАМ:\n"
    metrics_text += "="*60 + "\n"
    for ticker, result in ticker_results.items():
        metrics_text += (
            f"\n{ticker} (капитал: {ticker_capital[ticker]:,.0f} $, вес: {weights[ticker]*100:.1f}%):\n"
            f"  Final equity:  {result['final_equity_strategy']:,.0f} $\n"
            f"  Return:        {result['total_return_strategy']*100:,.1f} %\n"
            f"  CAGR:          {result['cagr_strategy']*100:,.2f} %\n"
            f"  Max DD:        {result['max_drawdown']*100:,.2f} %\n"
            f"  Sharpe:        {result['sharpe']:.2f}\n"
            f"  Экспозиция:    {result['market_exposure']:.1f}%\n"
            f"  Сделок:        {result['num_trades']}\n"
        )
    
    print("\n" + "="*80)
    print(metrics_text)
    print("="*80 + "\n")
    
    # Сохраняем метрики
    if save_metrics:
        #create counter to avoid overwriting
        counter = 1
        
        while os.path.exists(f"metrics_{counter}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt"):
            counter += 1
        filename = f"metrics_{counter}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(metrics_text)
        print(f"Метрики портфеля сохранены в {filename}")
    
    # --- ГРАФИК ПОРТФЕЛЯ ---
    if show_portfolio_plot:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Обработка нормализации по золоту
        portfolio_equity_plot = portfolio_equity.copy()
        portfolio_bh_equity_plot = portfolio_bh_equity.copy()
        ylabel_suffix = "($)"
        
        if normalize_by.lower() == 'gold' and 'GLD' in ticker_results:
            # Берем цены золота из ticker_results
            gld_data = ticker_results['GLD']['data']
            gld_analysis = gld_data[gld_data['is_analysis_period']]
            gld_prices = gld_analysis.loc[all_dates, 'Close']
            
            # Нормализуем equity по цене золота
            # Нормализуем так, чтобы первое значение портфеля равнялось первому значению GLD цены
            if len(gld_prices) > 0 and gld_prices.iloc[0] > 0:
                first_gld_price = gld_prices.iloc[0]
                portfolio_equity_plot = portfolio_equity / first_gld_price * 100  # В условных "унциях" (normalized)
                portfolio_bh_equity_plot = portfolio_bh_equity / first_gld_price * 100
                ylabel_suffix = "(oz GLD, normalized)"
        elif normalize_by.lower() == 'gold':
            print(f"⚠️ WARNING: Gold normalization requested but GLD not in tickers. Using USD.")
        
        # 1) Equity curves портфеля
        ax1 = axes[0]
        ax1.plot(portfolio_equity_plot.index, portfolio_equity_plot.values, 
                label="Portfolio Strategy", color="blue", linewidth=2)
        ax1.plot(portfolio_bh_equity_plot.index, portfolio_bh_equity_plot.values,
                label="Portfolio Buy & Hold", color="gray", linestyle="--", linewidth=2, alpha=0.7)
        
        ax1.set_title(f"Portfolio Equity Curve: {', '.join(tickers)} (Numeraire: {normalize_by.upper()})")
        ax1.set_ylabel(f"Equity {ylabel_suffix}")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # 2) Equity curves отдельных тикеров
        ax2 = axes[1]
        for ticker, result in ticker_results.items():
            data = result['data']
            analysis_data = data[data['is_analysis_period']]
            equity = analysis_data.loc[all_dates, 'equity_curve']
            ax2.plot(equity.index, equity.values, label=f"{ticker} ({weights[ticker]*100:.0f}%)", linewidth=1.5, alpha=0.8)
        
        ax2.set_title("Individual Ticker Equity Curves")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Equity ($)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')
        
        # Панель с метриками
        text_ax = fig.add_axes([0.78, 0.15, 0.2, 0.7])
        text_ax.axis("off")
        text_ax.text(0, 1, metrics_text, va="top", fontsize=9, family='monospace')
        
        plt.tight_layout(rect=[0, 0, 0.76, 1])
        plt.show()
    
    # Возвращаем результаты
    portfolio_metrics = {
        'tickers': tickers,
        'weights': weights,
        'start_date': start_date_actual,
        'end_date': end_date_actual,
        'years': years,
        'portfolio_equity': portfolio_equity,
        'portfolio_bh_equity': portfolio_bh_equity,
        'cagr_portfolio': cagr_portfolio,
        'cagr_bh': cagr_bh,
        'total_return_portfolio': total_ret_portfolio,
        'total_return_bh': total_ret_bh,
        'max_drawdown': max_dd,
        'sharpe': sharpe,
        'realized_portfolio_vol': realized_portfolio_vol,
        'target_vol': volatility_target,
        'leverage_applied': leverage_applied,
        'cash_rate': cash_rate,
        'avg_exposure': avg_exposure,
        'total_trades': total_trades,
        'ticker_results': ticker_results
    }
    
    return portfolio_metrics


