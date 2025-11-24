import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering

# Logging Configuration
logger = logging.getLogger(__name__)

def Banner():
    banner = (r"""
    =============================================
        🚀 Unsupervised ML Ensemble Model 🚀
            for Inventory Classification

            Author:    
            Andrés Felipe Toscano Serrano
          
            12/10/2025 - V.2
    =============================================
    """)
    print(banner)

def CostMatrix(df, item_id, holding_cost_columns, stockout_cost_columns, carrying_rate, fill_rate):
    """
    Matriz de costos por clase:
    C_stockout = (1 - alpha) * mu * c_s
    Holding: h_día * value * SS, con SS = z * sigma * sqrt(L).
    Devuelve columnas: COST_A/B/C (+ HOLD_* y STOCK_* si return_components=True)
    """
    # --------- mapeo de columnas ----------
    unit_cost_col, sigmaL_col = holding_cost_columns
    stockout_cost_col, mu_col = stockout_cost_columns

    required = [item_id, unit_cost_col, sigmaL_col, stockout_cost_col, mu_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")
    
    d = df[[item_id, unit_cost_col, sigmaL_col, stockout_cost_col, mu_col]]

    h_day = carrying_rate / 365
    classes = list(fill_rate.keys())

    # salida
    out = {item_id: d[item_id].values}

    hold_parts = {}
    stock_parts = {}

    for cls in classes:
        alpha = fill_rate[cls]
        z = float(norm.ppf(alpha))  # z-score del nivel de servicio

        # Safety Stock (usa directamente la desviación en lead time)
        SS = z * d[sigmaL_col].values

        # Holding cost
        hold = h_day * d[unit_cost_col].values * SS

        # Stockout cost (simple)
        stock = (1.0 - alpha) * d[mu_col].values * d[stockout_cost_col].values

        # Total
        out[f"COST_{cls}"] = hold + stock

        hold_parts[f"C_HOLD_{cls}"] = hold
        stock_parts[f"C_STOCKOUT_{cls}"] = stock

    df_cost_matrix = pd.DataFrame(out)

    for k, v in hold_parts.items():
        df_cost_matrix[k] = v
    for k, v in stock_parts.items():
        df_cost_matrix[k] = v

    return df_cost_matrix

def Cost_ComputeTags(row):

    # ABC
    if row['TAG_ABC'] == 'A':
        cost_abc = row[f"COST_A"]
    elif row['TAG_ABC'] == 'B':
        cost_abc = row[f"COST_B"]
    elif row['TAG_ABC'] == 'C':
        cost_abc = row[f"COST_C"]

    # KMEANS
    if row['TAG_KMEANS'] == 'A':
        cost_kmeans = row[f"COST_A"]
    elif row['TAG_KMEANS'] == 'B':
        cost_kmeans = row[f"COST_B"]
    elif row['TAG_KMEANS'] == 'C':
        cost_kmeans = row[f"COST_C"]

    # HIERARCHICAL
    if row['TAG_HC'] == 'A':
        cost_hc = row[f"COST_A"]
    elif row['TAG_HC'] == 'B':
        cost_hc = row[f"COST_B"]
    elif row['TAG_HC'] == 'C':
        cost_hc = row[f"COST_C"]

    # GMM
    if row['TAG_GMM'] == 'A':
        cost_gmm = row[f"COST_A"]
    elif row['TAG_GMM'] == 'B':
        cost_gmm = row[f"COST_B"]
    elif row['TAG_GMM'] == 'C':
        cost_gmm = row[f"COST_C"]
    
    # -------- 1) Caso: los tres modelos coinciden
    if row['TAG_KMEANS'] == row['TAG_HC'] == row['TAG_GMM']:
        
        tag_final = row['TAG_KMEANS']
        cost_final = row[f"COST_{tag_final}"]

        return pd.Series({
            'TAG_FINAL': tag_final,
            'COST_FINAL': cost_final,
            'COST_ABC': cost_abc,
            'COST_KMEANS': cost_kmeans,
            'COST_HC': cost_hc,
            'COST_GMM': cost_gmm
        })
    
    # -------- 2) Caso: no hay unanimidad → elegir menor costo
    costs = {
        'A': row['COST_A'],
        'B': row['COST_B'],
        'C': row['COST_C']
    }

    # Obtener el costo mínimo
    min_cost = min(costs.values())

    # Filtrar cuáles empatan con ese costo
    tied_tags = [tag for tag, cost in costs.items() if cost == min_cost]

    # Desempatar según prioridad A > B > C
    priority = {'A': 3, 'B': 2, 'C': 1}
    tag_final = max(tied_tags, key=lambda t: priority[t])

    cost_final = costs[tag_final]

    return pd.Series({
        'TAG_FINAL': tag_final, 
        'COST_FINAL': cost_final,
        'COST_ABC': cost_abc,
        'COST_KMEANS': cost_kmeans,
        'COST_HC': cost_hc,
        'COST_GMM': cost_gmm
    })

def Clustering_ComputeTags(row):

    # ABC
    if row['TAG_ABC'] == 'A':
        cost_abc = row[f"COST_A"]
    elif row['TAG_ABC'] == 'B':
        cost_abc = row[f"COST_B"]
    elif row['TAG_ABC'] == 'C':
        cost_abc = row[f"COST_C"]

    # KMEANS
    if row['TAG_KMEANS'] == 'A':
        cost_kmeans = row[f"COST_A"]
    elif row['TAG_KMEANS'] == 'B':
        cost_kmeans = row[f"COST_B"]
    elif row['TAG_KMEANS'] == 'C':
        cost_kmeans = row[f"COST_C"]

    # HIERARCHICAL
    if row['TAG_HC'] == 'A':
        cost_hc = row[f"COST_A"]
    elif row['TAG_HC'] == 'B':
        cost_hc = row[f"COST_B"]
    elif row['TAG_HC'] == 'C':
        cost_hc = row[f"COST_C"]

    # GMM
    if row['TAG_GMM'] == 'A':
        cost_gmm = row[f"COST_A"]
    elif row['TAG_GMM'] == 'B':
        cost_gmm = row[f"COST_B"]
    elif row['TAG_GMM'] == 'C':
        cost_gmm = row[f"COST_C"]
    
    max_value = max(row['SIL_KMEANS'], row['SIL_HC'], row['SIL_GMM'])

    if max_value == row['SIL_KMEANS']:
        tag_final = row['TAG_KMEANS']
        cost_final = row[f"COST_{tag_final}"]

    elif max_value == row['SIL_HC']:
        tag_final = row['TAG_HC']
        cost_final = row[f"COST_{tag_final}"]
    else:
        tag_final = row['TAG_GMM']
        cost_final = row[f"COST_{tag_final}"]

    return pd.Series({
        'TAG_FINAL': tag_final,
        'COST_FINAL': cost_final,
        'COST_ABC': cost_abc,
        'COST_KMEANS': cost_kmeans,
        'COST_HC': cost_hc,
        'COST_GMM': cost_gmm
    })


class Preprocess():
    def __init__(self, file_name: str, predictor_columns: list[str], with_pca: bool = True):
        self.file_name = file_name
        self.predictor_columns = predictor_columns
        self.with_pca = with_pca

        self.df = None
        self.eigenvalues_ = None
        self.n_components_ = None
        self.pca_ = None

    def _load_dataset(self):
        """Load a dataset from a CSV file."""
        try:    
            dataset_path = os.path.join(os.path.join(os.path.dirname(__file__), 'dataset'), self.file_name)
            df = pd.read_csv(dataset_path)
            self.df = df
            logger.info("✅ Dataset loaded correctly.")

            return df

        except Exception as e:
            logger.error(f"❌ Error loading the dataset: {e}")
            return None

    def _preprocess_data(self):
        """Preprocesses data, applying appropriate transformations."""
        df = self._load_dataset()
        try:
            if not all(col in df.columns for col in self.predictor_columns):
                raise ValueError("Some specified columns do not exist in the dataset.")
            
            X = df[self.predictor_columns]
            
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            numerical_cols = X.select_dtypes(include=['number']).columns
            
            transformer = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_cols),
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
                ]
            )

            # Pipeline
            pipeline = Pipeline([
                ("preprocessor", transformer)
            ])

            # 1) fit + transform del preprocesamiento base
            X_pre = pipeline.fit_transform(X)

            if self.with_pca:
                logger.info(f'✅ PCA :: {self.with_pca}')
                # 2) Estandarizar todo el espacio (para criterio de Kaiser)
                scaler = StandardScaler()
                X_std = scaler.fit_transform(X_pre)

                # 3) PCA
                pca = PCA()
                pca.fit(X_std)

                # 4) Elegir k por Kaiser (autovalor > 1)
                eigenvalues = pca.explained_variance_
                n_components = np.sum(eigenvalues > 1)
                if n_components == 0:
                    n_components = 1  # mínimo 1 componente

                # 🧾 Imprimir tabla simple de componentes y autovalores
                print("\n📊 Eigenvalues por componente:")
                for i, ev in enumerate(eigenvalues, start=1):
                    status = "✅ > 1" if ev > 1 else "❌ ≤ 1"
                    print(f"PC{i:02d}: {ev:.4f}  ({status})")

                # 5) Reajustar PCA con k componentes
                pca = PCA(n_components=n_components)
                X_pca = pca.fit_transform(X_std)

                # Guardar para inspección posterior
                self.eigenvalues_ = eigenvalues
                self.n_components_ = n_components
                self.pca_ = pca

                logger.info(f"✅ PCA applied: {n_components} components retained (eigenvalues > 1).")

                X_transformed = X_pca

            else:
                logger.info(f'❌ PCA :: {self.with_pca}')
                X_transformed = X_pre
            
            return df, X, X_transformed

        except Exception as e:
            logger.error(f"❌ Error in data preprocessing: {e}")
            return None, None, None
        
    def execute_preprocess_data(self) -> None:
        preprocess_data = self._preprocess_data()
        return preprocess_data

class EnsembleModel():
    def __init__(self, df, X, X_transformed, n_clusters, item_id, ranking_column, file_name, cost_matrix, priority, with_pca):
        self.df = df
        self.X = X
        self.X_transformed = X_transformed
        self.n_clusters = n_clusters
        self.item_id = item_id 
        self.ranking_column = ranking_column
        self.file_name = file_name
        self.cost_matrix = cost_matrix
        self.priority = priority
        self.with_pca = with_pca

        self.df_final = None
        self.df_meta_model = None


    def _train_base_learners(self, models):
        results = {}

        for name, model in models.items():
            logger.info(f"<<< ⚠️  :: Inventory Classification :: {name.upper()} >>>")

            if name.upper() == 'ABC':
                df_result = self.X.copy()
                tag_col = f"TAG_{name.upper()}"
                sil_col = f"SIL_{name.upper()}"
                df_result = df_result.sort_values(by=self.ranking_column, ascending=False).reset_index(drop=True)
                df_result['cum_pct'] = df_result[self.ranking_column].cumsum() / df_result[self.ranking_column].sum()

                def classify(p):
                    if p <= 0.80:
                        return 'A'
                    elif p <= 0.95:
                        return 'B'
                    else:
                        return 'C'

                df_result[tag_col] = df_result['cum_pct'].apply(classify)

                silhouette_avg = None
                labels = None
                df_result[sil_col] = silhouette_avg

                df_result = df_result.drop('cum_pct', axis=1)

                logger.info(f"<<< 🎯 :: Inventory Classification :: {name.upper()} >>>")
                class_counts = df_result[tag_col].value_counts().sort_index()
                tag_counts = {}
                for cluster in ['A', 'B', 'C']:
                    count = class_counts.get(cluster, 0)
                    perc = (count / len(df_result)) * 100
                    logger.info(f"  > Class {cluster}: {count} items ({perc:.2f}%)")
                    tag_counts[cluster] = {"count": count, "percentage": perc}

                # Añadir ID si existe
                if getattr(self, "item_id", None) and self.df is not None and self.item_id in self.df.columns:
                    df_result[self.item_id] = self.df[self.item_id].values

            else:
                # Entrenar modelo y obtener etiquetas
                if hasattr(model, "fit_predict"):
                    labels = model.fit_predict(self.X_transformed)
                else:
                    model.fit(self.X_transformed)
                    labels = getattr(model, "labels_", model.predict(self.X_transformed))

                # Calcular Silhouette Score
                silhouette_avg = silhouette_score(self.X_transformed, labels)
                logger.info(f"Model Parameters: {model.get_params() if hasattr(model, 'get_params') else 'N/A'}")
                logger.info(f"Silhouette Score: {silhouette_avg:.4f}")

                # Crear DataFrame de resultados
                df_result = self.X.copy()
                tag_col = f"TAG_{name.upper()}"
                sil_col = f"SIL_{name.upper()}"
                df_result[tag_col] = labels
                df_result[sil_col] = silhouette_avg

                # Ranking o tamaño
                if getattr(self, "ranking_column", None) and self.df is not None and self.ranking_column in self.df.columns:
                    df_result[self.ranking_column] = self.df[self.ranking_column].values
                    cluster_sum = df_result.groupby(tag_col)[self.ranking_column].sum().sort_values(ascending=False)
                else:
                    logger.warning("No valid column was specified for sorting. Cluster size will be used.")
                    cluster_sum = df_result.groupby(tag_col).size().sort_values(ascending=False)

                # Asignar etiquetas A, B, C...
                cluster_to_label = {cluster: chr(65 + i) for i, cluster in enumerate(cluster_sum.index)}
                df_result[tag_col] = df_result[tag_col].map(cluster_to_label)

                # Añadir ID si existe
                if getattr(self, "item_id", None) and self.df is not None and self.item_id in self.df.columns:
                    df_result[self.item_id] = self.df[self.item_id].values

                # Conteo
                label_counts = df_result[tag_col].value_counts().sort_index()
                label_percentages = (label_counts / label_counts.sum()) * 100

                logger.info(f"<<< 🎯 :: Inventory Classification :: {name.upper()} >>>")
                tag_counts = {}
                for cluster in label_counts.index:
                    count = label_counts[cluster]
                    perc = label_percentages[cluster]
                    logger.info(f"  > Class {cluster}: {count} items ({perc:.2f}%)")
                    tag_counts[cluster] = {"count": count, "percentage": perc}
                                
            logger.info(f'<<< ✅ :: Inventory Classification :: {name.upper()} >>>\n')
            logger.info(f'<<< ================================================== >>>\n')

            results[name] = {
                "model": model,
                "labels": labels,
                "silhouette": silhouette_avg,
                "tag_counts": tag_counts,
                "df_result": df_result
            }

        return results
    
    def _merge_results(self, results: dict) -> dict:
        logger.info(f"<<< ⚠️  :: Inventory Classification :: Merge Dataframes >>>")
        df_merged = None

        for name, data in results.items():
            df_model = data.get("df_result")
            if df_model is None or self.item_id not in df_model.columns:
                continue

            tag_cols = [c for c in df_model.columns if c.startswith("TAG_") or c.startswith("SIL_")]
            cols_to_merge = [self.item_id] + tag_cols

            if df_merged is None:
                df_merged = df_model[cols_to_merge].copy()
            else:
                df_merged = df_merged.merge(df_model[cols_to_merge], on=self.item_id, how="left")

        results["merge_models"] = {
            "model": 'MergeModels',
            "labels": None,
            "silhouette": None,
            "tag_counts": None,
            "df_result": df_merged
        }

        results["cost_matrix"] = {
            "model": 'cost_matrix',
            "labels": None,
            "silhouette": None,
            "tag_counts": None,
            "df_result": self.cost_matrix
        }

        results["preprocess"] = {
            "model": 'Preprocess',
            "labels": None,
            "silhouette": None,
            "tag_counts": None,
            "df_result": self.X_transformed
        }

        self.df_final = pd.merge(results["merge_models"]["df_result"], results["cost_matrix"]["df_result"], on=self.item_id, how='left')

        logger.info(f"<<< ✅ :: Inventory Classification :: Merge Dataframes >>>")
        logger.info(f'<<< ================================================== >>>\n')

        return results
    
    def _meta_model(self, results: dict) -> dict:
        
        model = json.dumps({
            "criterio": self.priority,
            "pca": self.with_pca
        })

        logger.info(f"<<< ⚠️  :: Inventory Classification :: Meta Model >>>")
        logger.info(f"Model Parameters: {model}")

        df_final = self.df_final

        if self.priority == 'costs':
            df_final[['TAG_FINAL', 'COST_FINAL', 'COST_ABC', 'COST_KMEANS', 'COST_HC', 'COST_GMM']] = df_final.apply(Cost_ComputeTags, axis=1)
        else:
            df_final[['TAG_FINAL', 'COST_FINAL', 'COST_ABC', 'COST_KMEANS', 'COST_HC', 'COST_GMM']] = df_final.apply(Clustering_ComputeTags, axis=1)

        sil_meta = None
        try:
            unique_tags = sorted(df_final['TAG_FINAL'].unique())
            tag_to_int = {tag: i for i, tag in enumerate(unique_tags)}
            final_labels = df_final['TAG_FINAL'].map(tag_to_int).values

            sil_meta = silhouette_score(self.X_transformed, final_labels)
            logger.info(f"Silhouette Score: {sil_meta:.4f}")
        except Exception as e:
            logger.warning(f"Error silhouette Score for MetaModel: {e}")
            sil_meta = None

        df_final['SIL_META'] = sil_meta


        df_final = df_final[['PRODUCT_ID','TAG_FINAL', 'COST_FINAL', 'SIL_META', 'TAG_ABC', 'COST_ABC', 'TAG_KMEANS', 'COST_KMEANS', 'SIL_KMEANS', 'TAG_HC', 'COST_HC', 'SIL_HC','TAG_GMM', 'COST_GMM', 'SIL_GMM']]

        self.df_meta_model = df_final

        results["meta_model"] = {
            "model": model,
            "labels": df_final['TAG_FINAL'].tolist(),
            "silhouette": sil_meta,
            "tag_counts": df_final['TAG_FINAL'].value_counts().to_dict(),
            "df_result": self.df_meta_model
        }

        logger.info(f"<<< ✅ :: Inventory Classification :: Meta Model >>>")
        logger.info(f'<<< ================================================== >>>\n')

        return results
    
    def _evaluation(self, results):

        models = {
            "ENSEMBLE":   ("TAG_FINAL",  "COST_FINAL"),
            "ABC":     ("TAG_ABC",    "COST_ABC"),
            "KMEANS":  ("TAG_KMEANS", "COST_KMEANS"),
            "HC":      ("TAG_HC",     "COST_HC"),
            "GMM":     ("TAG_GMM",    "COST_GMM"),
        }

        # Descubrimos el universo de tags presentes en cualquier modelo (A, B, C, etc.)
        all_tags = sorted(set().union(*[set(self.df_meta_model[tag].unique()) for tag, _ in models.values()]))

        summary_rows = []
        for model_name, (tag_col, cost_col) in models.items():
            # Conteo por TAG para este modelo
            counts = self.df_meta_model[tag_col].value_counts().to_dict()
            # Suma de costos de este modelo
            total_cost = self.df_meta_model[cost_col].sum()

            # Armamos la fila de resumen
            row = {"model": model_name, "total_cost": total_cost}
            # Garantizamos columnas de conteo para todos los tags (0 si falta)
            for t in all_tags:
                row[f"count_{t}"] = counts.get(t, 0)

            summary_rows.append(row)

        summary = pd.DataFrame(summary_rows)

        # Ordenado por menor costo
        summary_sorted = summary.sort_values("total_cost", ascending=True).reset_index(drop=True)

        # Identificar automáticamente todas las columnas de conteo (las que empiezan con 'count_')
        count_cols = [col for col in summary_sorted.columns if col.startswith('count_')]

        # Crear nueva columna con la suma de todos los conteos
        summary_sorted['Total_SKUs'] = summary_sorted[count_cols].sum(axis=1)

        print("\n📊 Summary Performance by Model:")
        print(summary_sorted.to_string(index=False))

        # Modelo con menor costo
        best_model = summary_sorted.loc[0, "model"]
        best_cost = summary_sorted.loc[0, "total_cost"]

        results["performance_evaluation"] = {
            "model": 'PerformanceEvaluation',
            "labels": None,
            "silhouette": None,
            "tag_counts": None,
            "df_result": summary_sorted
        }

        print(f"\n🥇 Modelo con menor costo total: {best_model} (total_cost = {best_cost})")

        return results

    def _export_results(self, results: dict):
        logger.info(f"<<< ⚠️  :: Inventory Classification :: Export Results >>>")
        today = datetime.now().strftime("%d_%m_%Y")

        base_dir = Path(__file__).resolve().parent

        dataset_name = Path(self.file_name).stem

        output_dir =base_dir / "results" / today / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, data in results.items():
            model_dir = output_dir / name
            model_dir.mkdir(exist_ok=True)

            # --- Preparar resumen JSON ---
            silhouette = data.get("silhouette")
            labels = data.get("labels")
            model = data.get("model")
            df_result = data.get("df_result")
            tag_counts = data.get("tag_counts")

            if name == 'meta_model':
                model_params = model
            elif hasattr(model, "get_params"):
                model_params = model.get_params()
            else:
                model_params = None

            summary = {
                "tag_counts": (
                    tag_counts
                ),
                "model_params": (
                    model_params
                ),
                "silhouette": (
                    float(silhouette) if silhouette is not None else None
                ),
                "labels": (
                    labels.tolist() if hasattr(labels, "tolist") else (labels if labels is not None else None)
                ),
                "n_samples": len(labels) if labels is not None else None
            }

            json_path = model_dir / f"{name}_summary.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=4, ensure_ascii=False, 
                        default=lambda o: float(o) if isinstance(o, (np.floating, float)) 
                        else int(o) if isinstance(o, (np.integer, int)) 
                        else str(o))

            # --- Exportar DataFrame si existe ---
            if isinstance(df_result, pd.DataFrame):
                csv_path = model_dir / f"{name}_df.csv"
                df_result.to_csv(csv_path, index=False)

            logger.info(f"✅ Results exported for '{name}' in {model_dir.resolve()}")

        logger.info(f"<<< ✅  :: Inventory Classification :: Export Results >>>")
        logger.info(f'<<< ================================================== >>>\n')

    def execute_ensemble(self) -> None:
        # >> Models 
        models = {
            "kmeans": KMeans(n_clusters=self.n_clusters, random_state=42),
            "hc": AgglomerativeClustering(n_clusters=self.n_clusters),
            "gmm": GaussianMixture(n_components=self.n_clusters, random_state=42),
            "abc": 'ABC'
        }
        results = self._train_base_learners(models=models)

        results_merge = self._merge_results(results=results)

        results_meta_model = self._meta_model(results=results_merge)

        results_ensemble = self._evaluation(results=results_meta_model)

        self._export_results(results_ensemble)