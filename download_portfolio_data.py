"""
Скрипт для скачивания данных нескольких тикеров
Каждый тикер сохраняется в свой файл: {ticker}.csv
"""
from data_download import download_data
import main as config
# Список тикеров для скачивания
tickers = config.portfolio_tickers

# Период данных
start_date = "1999-01-01"
end_date = "2025-01-01"

if __name__ == "__main__":
    print("\n" + "="*80)
    print("СКАЧИВАНИЕ ДАННЫХ ДЛЯ ПОРТФЕЛЯ")
    print("="*80 + "\n")
    
    for ticker in tickers:
        try:
            download_data(ticker, start_date, end_date)
        except Exception as e:
            print(f"❌ Ошибка при скачивании {ticker}: {e}\n")
            continue
    
    print("\n" + "="*80)
    print("✓ СКАЧИВАНИЕ ЗАВЕРШЕНО")
    print("="*80)
