
import logging
from pathlib import Path

from utils import Banner, Preprocess, CostMatrix, EnsembleModel

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    # =======================================================
    #               Inputs
    # =======================================================
    # >> Datasets Paths
    data_dir = Path(__file__).resolve().parent / "dataset"

    dataset_files = data_dir.glob("retailer_*.csv")

    if not dataset_files:
        print("⚠️  No datasets were found with the pattern 'retailer_*.csv'")
        return
        # All Variables 
            # UNITS, ORDERS, SALES
            # UNITS_STOCKOUT, ORDERS_STOCKOUT, UNITS_STOCKOUT_COST
            # UNITS_DEMAND, AVG_DAILY_DEMAND, STDDEV_DEMAND_FINAL
            # UNIT_COST, AVG_STOCK_LEVEL, MEDIAN_STOCK_LEVEL
            # LEAD_TIME_DAYS, STDDEV_LEAD_TIME

    # >> Parameters
    n_clusters = 3
    item_id = "PRODUCT_ID"
    ranking_column = "ORDERS"
    predictor_columns = ["UNITS", "ORDERS", "LEAD_TIME_DAYS", "AVG_STOCK_LEVEL"]

    holding_cost_columns = ["UNIT_COST", "STDDEV_LEAD_TIME"]
    stockout_cost_columns = ["UNITS_STOCKOUT_COST", "AVG_DAILY_DEMAND"]
    carrying_rate=0.25
    fill_rate={
        "A": 0.98,
        "B": 0.95,
        "C": 0.90
        }
    # =======================================================




    # =======================================================
    #              Step by Step
    # =======================================================
    for dataset_path in dataset_files:
        dataset_name = dataset_path.name
        print("\n" + "=" * 60)
        print(f"🚀 Procesando dataset: {dataset_name}")
        print("=" * 60)

        try:
            # >> Step 1: Load and Preprocess Data
            pre = Preprocess(
                file_name=str(dataset_path),
                predictor_columns=predictor_columns,
                with_pca=True
            )
            df, X, X_transformed = pre.execute_preprocess_data()

            # >> Step 2: Cost Matrix
            df_cost_matrix = CostMatrix(
                df=df,
                item_id=item_id,
                holding_cost_columns=holding_cost_columns, 
                stockout_cost_columns=stockout_cost_columns, 
                carrying_rate=carrying_rate, 
                fill_rate=fill_rate
            )
            
            # >> Step 3: Ensemble Model
            em = EnsembleModel(
                df=df,
                X=X,
                X_transformed=X_transformed,
                n_clusters=n_clusters,
                item_id=item_id,
                ranking_column=ranking_column, 
                file_name=dataset_name, 
                cost_matrix=df_cost_matrix
            )
            
            em.execute_ensemble()

            logging.info(f">>>>> ✅ Dataset {dataset_name} successfully processed and exported.")
            logging.info(f'<<< ================================================== >>>\n')

        except Exception as e:
            print(f"❌ Error processing {dataset_name}: {e}\n")

if __name__ == "__main__":
    Banner()
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Initializing Model.')
    main()
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Done.')