#!/usr/bin/env python3
# coding: utf-8
"""
Classes for nutritionist.py.
Expected files in ./data:
- model_final.pkl
- ingredients_columns.csv
- nutrition_facts.csv
- similar_recipes.csv
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


CLASS_NAMES = {
    0: "bad",
    1: "great",
    2: "so-so",
    "0": "bad",
    "1": "great",
    "2": "so-so",
}


class MissingIngredientsError(ValueError):
    """Raised when user entered ingredients that are absent in the ingredient database."""

    def __init__(self, missing: List[str]):
        self.missing = missing
        super().__init__(
            "The following ingredients are missing in our database: "
            + ", ".join(missing)
        )


class IngredientVectorizer:
    """Turns a user ingredient list into a one-row binary DataFrame."""

    def __init__(self, ingredients_columns_path = "data/ingredients_columns.csv"):
        if not os.path.exists(ingredients_columns_path):
            raise FileNotFoundError(
                f"File not found: {ingredients_columns_path}. Save the training ingredient columns first."
            )

        columns_df = pd.read_csv(ingredients_columns_path)
        if "ingredient" in columns_df.columns:
            self.columns = columns_df["ingredient"].astype(str).tolist()
        else:
            self.columns = columns_df.iloc[:, 0].astype(str).tolist()

        self._normalized_to_original = {
            self.normalize_name(col): col for col in self.columns
        }

    @staticmethod
    def normalize_name(name: str) -> str:
        name = name.lower().strip()
        name = re.sub(r"[^a-zA-Z\s-]", " ", name)
        name = re.sub(r"\s+", " ", name)
        return name.strip()

    @staticmethod
    def parse_ingredients(raw_ingredients: str | Iterable[str]) -> List[str]:
        if isinstance(raw_ingredients, str):
            items = raw_ingredients.split(",")
        else:
            items = list(raw_ingredients)
        return [item.strip().lower() for item in items if str(item).strip()]

    def get_missing(self, raw_ingredients: str | Iterable[str]) -> List[str]:
        user_ingredients = self.parse_ingredients(raw_ingredients)
        missing = []

        for ingredient in user_ingredients:
            normalized = self.normalize_name(ingredient)
            if normalized not in self._normalized_to_original:
                missing.append(ingredient)

        return missing

    def validate_ingredients(self, raw_ingredients: str | Iterable[str]) -> None:
        missing = self.get_missing(raw_ingredients)
        if missing:
            raise MissingIngredientsError(missing)

    def transform(self, raw_ingredients: str | Iterable[str]) -> pd.DataFrame:
        self.validate_ingredients(raw_ingredients)
        user_ingredients = self.parse_ingredients(raw_ingredients)
        row = pd.Series(0, index=self.columns, dtype=np.int8)

        for ingredient in user_ingredients:
            normalized = self.normalize_name(ingredient)
            row[self._normalized_to_original[normalized]] = 1

        return pd.DataFrame([row], columns=self.columns)


class RatingPredictor:
    """Loads the trained model and predicts bad / so-so / great."""

    def __init__(
        self,
        model_path: str = "data/model_final.pkl",
        ingredients_columns_path: str = "data/ingredients_columns.csv",
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"File not found: {model_path}. Train and save the final model first."
            )
        self.model = joblib.load(model_path)
        self.vectorizer = IngredientVectorizer(ingredients_columns_path)

    def predict(self, raw_ingredients: str | Iterable[str]) -> str:
        X = self.vectorizer.transform(raw_ingredients)
        pred = self.model.predict(X)[0]
        return CLASS_NAMES.get(pred, str(pred))


class NutritionFactsProvider:
    """Returns nutrition facts in % of Daily Value for ingredients."""

    def __init__(self, csv_path: str = "data/nutrition_facts.csv"):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"File not found: {csv_path}. Generate nutrition facts first."
            )
        self.df = pd.read_csv(csv_path)

        first_col = self.df.columns[0]
        if first_col.lower() not in {"ingredient", "ingredients"}:
            self.df = self.df.rename(columns={first_col: "ingredient"})
        else:
            self.df = self.df.rename(columns={first_col: "ingredient"})

        self.df["ingredient_norm"] = self.df["ingredient"].astype(str).str.lower().str.strip()

    def get(self, ingredient: str) -> Dict[str, float]:
        ingredient_norm = ingredient.lower().strip()
        row = self.df[self.df["ingredient_norm"] == ingredient_norm]
        if row.empty:
            return {}

        result = row.drop(columns=["ingredient", "ingredient_norm"]).iloc[0]
        result = pd.to_numeric(result, errors="coerce").fillna(0)
        return {nutrient: float(value) for nutrient, value in result.items() if value > 0}


class SimilarRecipesFinder:
    """Finds top-N similar recipes using binary ingredient columns."""

    def __init__(
        self,
        recipes_path: str = "data/similar_recipes.csv",
        ingredients_columns_path: str = "data/ingredients_columns.csv",
    ):
        if not os.path.exists(recipes_path):
            raise FileNotFoundError(
                f"File not found: {recipes_path}. Create the similar recipes CSV first."
            )

        self.recipes = pd.read_csv(recipes_path)
        self.vectorizer = IngredientVectorizer(ingredients_columns_path)

        required = {"title", "rating", "url"}
        missing = required - set(self.recipes.columns)
        if missing:
            raise ValueError(f"similar_recipes.csv must contain columns: {sorted(missing)}")

        self.ingredient_columns = [
            col for col in self.vectorizer.columns if col in self.recipes.columns
        ]
        if not self.ingredient_columns:
            raise ValueError("No ingredient columns found in similar_recipes.csv")

        self.recipe_matrix = self.recipes[self.ingredient_columns].fillna(0).astype(int)

    def find_top(self, raw_ingredients: str | Iterable[str], n: int = 3) -> List[Dict[str, object]]:
        query = self.vectorizer.transform(raw_ingredients)[self.ingredient_columns]

        if query.sum(axis=1).iloc[0] == 0:
            return []

        #distances = pairwise_distances(query, self.recipe_matrix, metric="jaccard")[0]
        distances = pairwise_distances(
            query.to_numpy(dtype=bool),
            self.recipe_matrix.to_numpy(dtype=bool),
            metric="jaccard"
        )[0]

        similarities = 1 - distances
        top_idx = np.argsort(similarities)[::-1][:n]

        results = []
        for idx in top_idx:
            row = self.recipes.iloc[idx]
            results.append(
                {
                    "title": row["title"],
                    "rating": row["rating"],
                    "url": row["url"],
                    "similarity": round(float(similarities[idx]), 3),
                }
            )
        return results
