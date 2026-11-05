# DSB12: Nutritionist CLI 🎓

What it's about
---------------
This project is a CLI tool that acts as a virtual nutritionist. It takes a list of ingredients, predicts the likely rating of a dish made from them ("bad", "so-so", or "great"), provides nutrition facts, and finds similar recipes. The core of the project is a machine learning model trained on the Epicurious Recipes dataset.

This was a collaborative project. My primary contribution was the end-to-end development of the machine learning pipeline in `recipes.ipynb`, including EDA, model training, and evaluation.

Stack
-----
- **Data Manipulation & EDA:** Pandas, NumPy, Matplotlib, Seaborn
- **Modeling:** Scikit-learn (Linear Models, SVC, Decision Trees, Random Forest, Ensembles)
- **Utilities:** `joblib` for model serialization, `tqdm` for progress bars

Results
-------
The initial regression task proved to be challenging, as linear models performed only slightly better than a naive prediction (R² ≈ 0.10). The problem was then reformulated as a binary classification task (predicting a "great" (4-5) or "bad" (0-3) rating).

The best-performing model was a **Random Forest Classifier** with `class_weight='balanced'`. This model's performance on the test set was:

| Metric      | Score  |
|-------------|--------|
| **F1-score**  | 0.788  |
| **Accuracy**  | 0.802  |

The model significantly outperforms the naive baseline, which simply predicts the most frequent class (F1 ≈ 0.885 was achieved by simply predicting all as "great", but the F1-score for the minority "bad" class was 0, highlighting the need for a balanced model).

Pipeline
--------
1.  **Data Loading & EDA:** The Epicurious dataset (`epi_r.csv`) was loaded, and ingredient columns were selected. The distribution of ratings was analyzed.
2.  **Problem Framing:**
    -   Initially, a regression task was attempted to predict the exact `rating` (R², RMSE).
    -   Due to poor performance, the task was changed to classification. The target was first rounded to the nearest integer and then binarized: ratings [0, 1, 2, 3] were labeled as `bad` (0), and [4, 5] as `great` (1).
3.  **Model Training & Selection:** Several models were trained and evaluated:
    -   Logistic Regression (with GridSearchCV)
    -   Support Vector Classifier (SVC)
    -   Decision Tree Classifier
    -   Random Forest Classifier
    -   Ensemble methods (Bagging, Stacking, Voting)
4.  **Final Model:** A `RandomForestClassifier` with `class_weight='balanced'` showed the best F1-score for the minority class and the best overall weighted F1-score. It was saved to a `.pkl` file for deployment.
5.  **Application:** The CLI tool (`nutritionist.py`) loads the trained model and other data files (`recipes.py` contains the helper classes) to provide the final functionality.

Lessons learned
---------------
-   **Problem Formulation is Key:** The low R² of linear regression models indicated that a regression approach was not suitable for this data. Reframing the problem as a binary classification task was a critical step that led to a much more useful and accurate model.
-   **Handling Class Imbalance:** The dataset is heavily skewed towards positive ratings. Using `class_weight='balanced'` in the Random Forest model was crucial for improving its ability to identify the minority "bad" rating class, which is often more interesting from a business perspective.
-   **Beyond Simple Models:** While linear models struggled, ensemble methods like Random Forest proved powerful for capturing the complex interactions between ingredients.
-   **Collaboration is a Skill:** This project highlighted the importance of communication and division of labor, which are as critical as technical skills.

How to run
----------
This project requires a set of data files and the trained model. After cloning the repository, you can run the CLI tool from the root directory:

```bash
python nutritionist.py "milk, honey, jam"
