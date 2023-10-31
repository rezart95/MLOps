from __future__ import annotations

import argparse
import itertools
import logging
import pathlib
import random
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import pulp
import statsmodels.api as sm
from azureml.core import Run
from azureml.core.model import Model
from numpy import ndarray

from helpers import fitted_vs_actuals_plot, mad, mape, residuals_plot

# ignore FutureWarnings from the statsmodels package's usage of pandas
warnings.simplefilter(action="ignore", category=FutureWarning)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(name)s]: %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("model.model")


class PromotionModel:
    def __init__(self, *, run: Run = None) -> None:
        self.run = run or Run.get_context()
        self.params = None
        self.ols_models = dict()
        self.lp = None
        self.solution = None

    def fit(self, df: pd.DataFrame) -> None:
        self.params = df.groupby("cat").apply(self.get_params, log=True)
        self.solution = self.optimize_promo(self.params)

    def _remove_data(self) -> None:
        for model in self.ols_models.values():
            model.remove_data()
        self.run = None

    def save(self, filename: str = "output/model.joblib") -> None:
        pathlib.Path(filename).parent.mkdir(parents=True, exist_ok=True)
        self._remove_data()
        joblib.dump(self, filename)

    @staticmethod
    def load(path) -> PromotionModel:
        return joblib.load(path)

    # Model responsiveness of each category to ads/discounts
    def get_params(self, data: pd.DataFrame, *, log: bool = False) -> ndarray:
        """        
        Calculate promotion parameters for single category        
        :param data: the row of the dataframe of categories and trasaction results        
        :type data: pd.dataframe        
        :return: the regression parameter estimates a single row        
        :rtype: numpy.ndarray        
        """
        y = data["profit"]
        x = data[["discounts", "ads"]]
        x = sm.add_constant(x)
        ols = sm.OLS(y, x)
        result = ols.fit()
        params = result.params[["const", "discounts", "ads"]]
        if log:
            counts, edges = np.histogram(result.resid, bins="auto")
            self.run.log_residuals(
                name=f"{data.name} Residuals",
                value={
                    "schema_type": "residuals",
                    "schema_version": "1.0.0",
                    "data": {
                        "bin_edges": edges.tolist(),
                        "bin_counts": counts.tolist(),
                    },
                },
            )
            self.run.log_row(
                name="OLS results",
                description="OLS results",
                category=data.name,
                const=params.const,
                discounts_coeff=params.discounts,
                ads_coeff=params.ads,
                aic=result.aic,
                bic=result.bic,
                llf=result.llf,
                rsquared=result.rsquared,
                rsquared_adj=result.rsquared_adj,
                mape=mape(y, result.fittedvalues),
                mad=mad(y, result.fittedvalues),
            )
            self.run.log_image(
                name=f"{data.name} Residuals",
                plot=residuals_plot(result.resid, data.name),
            )
            self.run.log_image(
                name=f"{data.name} Fitted vs Actuals",
                plot=fitted_vs_actuals_plot(result.fittedvalues, y, data.name),
            )

            self.ols_models[data.name] = result

        return params

    # Optimize based on the parameters    
    def optimize_promo(self, promo: pd.DataFrame) -> pd.DataFrame:        
        """        
        Optimizes ads and discounts for each category        
        :param promo: Parameter estimates from promotion regression model        
        :type promo: pd.dataframe        
        :return: the optimum number of ads and discount rate for each category        
        :rtype: pd.dataframe        
        """        
        
        lp = pulp.LpProblem("opt", pulp.LpMinimize)        
        
        # Create problem variables        
        discounts = [            
            pulp.LpVariable(str(promo.index[i] + "_d"), 0, 0.3, cat="Continuous")            
            for i in range(len(promo))        
        ]        
        ads = [            
            pulp.LpVariable(str(promo.index[i] + "_a"), 0, 3, cat="Integer")            
            for i in range(len(promo))        
        ]        
        
        d_rates = promo["discounts"].values        
        a_rates = promo["ads"].values        
        
        # lp to solve        
        lp += -(            
            sum(d * dr for d, dr in zip(discounts, d_rates))            
            + sum(a * ar for a, ar in zip(ads, a_rates))        
        )        
        
        # Aribtrarily diversifying the promotion        
        for i, j, k in itertools.permutations(ads, 3):            
            lp += pulp.lpSum([i, j, k]) <= 7        
        
        for i, j, k in itertools.permutations(discounts, 3):            
            lp += pulp.lpSum([i, j, k]) <= 0.7        
        
        # Arbitrarily limiting the ads per day        
        lp += pulp.lpSum(ads) <= len(ads) + 3

        lp.solve()        
        self.run.log(            
            name="LP Objective Function",            
            value=pulp.value(lp.objective),            
            description="Value of the LP objective function",        
        )        
        
        # Feed solution into dataframe        
        solution = [(v.name, v.varValue) for v in lp.variables()]        
        
        # log optimal solution parameters        
        for var_name, var_value in solution:            
            self.run.log_row(                
                name="LP Solution",                
                description="Optimal solution parameters",                
                var_name=var_name,                
                var_value=var_value,            
            )        
            
        return pd.DataFrame.from_records(solution, columns=["var", "val"])    
        
    # Evaluation metric is the lift over no promotion activity    
    def evaluate_promo(self) -> float:        
        """        
        Evaluates optimized promotion values based on lift compared to no promotion        
        :return: the lift that optimal promotion provides over no promotion        
        :rtype: float        
        """        
        no_promo_profit = sum(self.params["const"])        
        
        ad_params = self.solution[self.solution["var"].str[-1] == "a"]        
        dis_params = self.solution[self.solution["var"].str[-1] == "d"]

        ad_profit = sum(ad_params["val"].values * self.params["ads"].values)        
        dis_profit = sum(dis_params["val"].values * self.params["discounts"].values)

        promo_profit = no_promo_profit + ad_profit + dis_profit

        promo_lift = promo_profit / no_promo_profit

        return promo_lift

    # Generate preditced profits using promotion parameters    
    def pred_profit(self, df: pd.DataFrame, params: pd.DataFrame) -> pd.DataFrame:        
        """        
        Makes profit predictions using promotion parameters        
        :param df: the dataframe of categories and trasaction results        
        :type df: pd.dataframe        
        :param params: the promotion parameters from `get_params_df`        
        :type params: pd.dataframe        
        :return: the input df with additional column of predicted profit        
        :rtype: pd.dataframe        
        """        
        params.columns = ["constant", "discounts_params", "ads_params"]        
        df_p = pd.merge(df, params, left_on="cat", right_index=True)  

        df_p["pred"] = (            
            df_p["constant"]            
            + (df_p["discounts"] * df_p["discounts_params"])            
            + (df_p["ads"] * df_p["ads_params"])        
        )        
        
        df_p.drop(["constant", "discounts_params", "ads_params"], axis=1, inplace=True)        
        
        return df_p
        

def parse_args():    
    parser = argparse.ArgumentParser()    
    parser.add_argument("--input-dataset", dest="input_dataset", required=True)    
    parser.add_argument("--seed", dest="seed", required=False, default=755, type=int)   
    
    return parser.parse_args()
    

if __name__ == "__main__":    
    args = parse_args()    
    
    # set seed for reproducibility    
    random.seed(args.seed)    
    
    with Run.get_context() as run:        
        # Read in data        d
        df = pd.read_csv(args.input_dataset)        
        training_data = df[["cat", "discounts", "ads", "profit"]].copy(deep=True)        
        
        # Get optimal promotion        
        model = PromotionModel(run=run)        
        model.fit(training_data)        
        logger.info("\nOptimal Promotion:\n%s", model.solution)        
        
        # Evaluate optimized promotion        
        lift = model.evaluate_promo()       
        run.log(            
            name="Model Lift",            
            value=lift,            
            description="Lift of optimized promotion vs. no promotion",        
        )        
        
        # serialize and save the model        
        model.save("outputs/model.joblib")
