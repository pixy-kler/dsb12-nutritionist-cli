#!/usr/bin/env python3
# coding: utf-8

import sys
from typing import List

from recipes import (
    RatingPredictor,
    NutritionFactsProvider,
    SimilarRecipesFinder,
    IngredientVectorizer,
    MissingIngredientsError,
)


def print_forecast(predicted_class: str, ingredients: List[str]) -> None:
    joined = ", ".join(ingredients)
    print("I. OUR FORECAST\n")
    print(
        f"You might find it tasty, but in our opinion, it is a {predicted_class} idea "
        f"to have a dish with that list of ingredients: {joined}."
    )


def print_nutrition(ingredients: List[str], provider: NutritionFactsProvider) -> None:
    print("\nII. NUTRITION FACTS\n")
    for ingredient in ingredients:
        facts = provider.get(ingredient)
        print(ingredient.capitalize())
        if not facts:
            print("No nutrition data available")
        else:
            for nutrient, pct in facts.items():
                print(f"{nutrient} - {pct:g}% of Daily Value")
        print()


def print_similar_recipes(recipes: List[dict]) -> None:
    print("III. TOP-3 SIMILAR RECIPES:\n")
    if not recipes:
        print("No similar recipes found for this list of ingredients.")
        return

    for recipe in recipes:
        print(f"- {recipe['title']}, rating: {recipe['rating']}, URL:")
        print(recipe["url"])
        print()


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: ./nutritionist.py "milk, honey, jam"')
        sys.exit(1)

    raw_query = " ".join(sys.argv[1:])
    ingredients = IngredientVectorizer.parse_ingredients(raw_query)

    if not ingredients:
        print('Usage: ./nutritionist.py "milk, honey, jam"')
        sys.exit(1)

    try:
        predictor = RatingPredictor()
        predictor.vectorizer.validate_ingredients(ingredients)

        nutrition_provider = NutritionFactsProvider()
        similar_finder = SimilarRecipesFinder()

        predicted_class = predictor.predict(ingredients)
        top_recipes = similar_finder.find_top(ingredients, n=3)

    except MissingIngredientsError as error:
        print(error)
        sys.exit(1)

    print_forecast(predicted_class, ingredients)
    print_nutrition(ingredients, nutrition_provider)
    print_similar_recipes(top_recipes)


if __name__ == "__main__":
    main()
