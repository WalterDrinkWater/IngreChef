import pandas as pd
import  unidecode
import ast
import config

def get_recommendations(N, scores):
    df_recipes = pd.read_csv(config.RECIPES_DETAILS)
    top = sorted(range(len(scores)), key=lambda i: scores[i],reverse=True)[:N]
    recommendation = pd.DataFrame(columns=["recipe", "ingredients", "score"])

    count = 0
    for i in top:
        recommendation.at[count, "recipe"] = title_parser(df_recipes["recipe_name"][i])
        recommendation.at[count, "ingredients"] = ingredient_parser_final(df_recipes["ingredients"][i])
        recommendation.at[count, "score"] = f"{scores[i]}"
        count += 1
    return recommendation

def title_parser(title):
    title = unidecode.unidecode(title)
    return title

def ingredient_parser_final(ingredient):
    if isinstance(ingredient, list):
        ingredients = ingredient
    else:
        ingredients = ast.literal_eval(ingredient)

    ingredients = ",".join(ingredients)
    ingredients = unidecode.unidecode(ingredients)
    return ingredients