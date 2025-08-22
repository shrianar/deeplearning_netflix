# Deep Learning-Based Personalized Recommendation System for Netflix
A deep learning-based recommendation system using the Netflix Prize dataset. This project implements Neural Collaborative Filtering (NCF) and compares it against baseline models such as Singular Value Decomposition (SVD) and Alternating Least Squares (ALS). The methodology includes preprocessing, exploratory data analysis, model training, and evaluation with RMSE and MAE metrics.  

---

## Objectives
- Preprocess and clean the Netflix Prize dataset (~100M ratings).  
- Engineer features such as **ratings per user**, **ratings per movie**, and **average rating per movie**.  
- Implement **Neural Collaborative Filtering (NCF)** using PyTorch.  
- Compare against **SVD** (Surprise library) and **ALS** (Implicit library).  
- Evaluate models with **RMSE** and **MAE** metrics.  
- Visualize distributions, user activity, popularity bias, and model performance.  
- Explore future directions with **AutoRec** and **BERT4Rec**.  

---

## Methodology

### Data Preprocessing
- Filtered out users with fewer than **25 ratings** to reduce sparsity.  
- Saved ratings and metadata in **Parquet format** using PyArrow/FastParquet.  
- Cleaned and merged **movie metadata** (`movie_titles.csv`) with the ratings dataset.  
- Fixed incomplete or inconsistent **date fields**, converted to datetime.  
- Engineered new features:  
  - `rating_count_per_user` (total ratings by user)  
  - `rating_count_per_movie` (total ratings for a movie)  
  - `average_movie_rating` (mean rating per movie)  
- Applied **log transformations** to skewed features.  
- Created a **time-aware train/test split**:  
  - Training = ratings before 2005  
  - Testing = ratings from 2005 onward  
- Final dataset: **3.07M training records**, **2.73M testing records**.  

---

### Exploratory Data Analysis (EDA)
- Ratings distribution is **skewed toward 3–4 stars** (positive bias).  
- User activity is **long-tailed**: most users rate few movies, some rate thousands.  
- Movie popularity is **long-tailed**: a few movies dominate the rating counts.  
- Generated plots for:  
  - Distribution of movie ratings  
  - User activity distribution (log scale)  
  - Movie popularity distribution (log scale)  

<img width="296" height="476" alt="image" src="https://github.com/user-attachments/assets/79dae690-db67-41cb-ab8c-42d0fd057e43" />

---

### Models and Experiments

#### Neural Collaborative Filtering (NCF)
- Implemented in **PyTorch**.  
- Architecture: user embedding + item embedding → concatenation → multi-layer perceptron → predicted rating.  
- Loss: MSELoss  
- Optimizer: Adam  
- Results:  
  - RMSE = **1.073**  
  - MAE = **0.869**  
- Scatterplot showed close alignment between predicted and true ratings.  

<img width="240" height="151" alt="image" src="https://github.com/user-attachments/assets/8e27b65d-5af2-4760-b8a4-cc63fe6de0a6" />

---

#### Singular Value Decomposition (SVD)
- Implemented with the **Surprise library**.  
- Baseline matrix factorization model.  
- Results:  
  - RMSE = **1.018**  
  - MAE = **0.814**  
- Performed slightly better than NCF, confirming the strength of SVD as a baseline.  

---

#### Alternating Least Squares (ALS)
- Implemented with the **Implicit library**.  
- Optimized for implicit feedback, but tested here on explicit ratings.  
- Results:  
  - RMSE = **3.68**  
- ALS performed poorly in this setting compared to NCF and SVD.  

---

### Model Comparison
| Model | RMSE   | MAE   |
|-------|--------|-------|
| NCF   | 1.073  | 0.869 |
| SVD   | 1.018  | 0.814 |
| ALS   | 3.680  | —     |

<img width="295" height="166" alt="image" src="https://github.com/user-attachments/assets/fe245300-f9f6-4147-9934-fcfaf2514a5c" />


---

## Conclusion
- **NCF** captured complex latent interactions and was competitive with SVD.  
- **SVD** outperformed NCF slightly, showing the continued strength of traditional matrix factorization.  
- **ALS** was unsuitable for explicit rating prediction, performing poorly.  
- The project demonstrates that deep learning methods can rival classical methods and provide flexibility for future extensions.  

**Future Work**:  
- Implement **AutoRec** (autoencoder-based collaborative filtering).  
- Explore **BERT4Rec** (Transformer-based sequential recommendation).  
- Evaluate ranking-based metrics (Precision@K, Recall@K, NDCG).  

---

## References
- He, X. et al. (2017). Neural Collaborative Filtering.  
- Sedhain, S. et al. (2015). AutoRec: Autoencoders Meet Collaborative Filtering.  
- Sun, F. et al. (2019). BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations.  
- Netflix Prize Dataset ([kaggle](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)).  

---

## License
Include a license of your choice (e.g., MIT) in the `LICENSE` file.
