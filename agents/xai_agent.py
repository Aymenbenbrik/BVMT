import torch
import numpy as np

class XAIAgent:
    """
    Explainability Agent (XAI).
    Extracts SHAP values and attention weights from existing models to provide
    interpretable narratives for predictions.
    """
    def __init__(self, db_url: str = None):
        self.db_url = db_url
        print("[XAIAgent] Initialized")
    
    async def run(self, state) -> dict:
        """
        Extract explanations from all successful prior agent runs.
        """
        xai_result = {
            'fundamental': None,
            'technical': None,
            'graph': None
        }
        
        # 1. Execute Fundamental Explanations (SHAP)
        if not state.skip_fundamental and 'fundamental' in state.agent_outputs:
            if 'key_ratios' in state.agent_outputs['fundamental']:
                xai_result['fundamental'] = self._explain_fundamental(
                    features=state.agent_outputs['fundamental']['key_ratios'],
                    model_path='models/fundamental_xgb.pkl'
                )

        # 2. Execute Technical Explanations (TFT attention)
        if 'technical' in state.agent_outputs and 'xai_attention' in state.agent_outputs['technical']:
            xai_result['technical'] = state.agent_outputs['technical']['xai_attention']
        else:
            xai_result['technical'] = {
                "note": "TFT interpretability not available in output."
            }

        # 3. Execute Graph Explanations
        if 'graph' in state.agent_outputs and 'graph_view_3d' in state.agent_outputs['graph']:
            xai_result['graph'] = self._explain_graph_summary(
                state.agent_outputs['graph']
            )

        state.agent_outputs['xai'] = xai_result
        return state

    def _explain_fundamental(self, features: dict, model_path: str) -> dict:
        """
        Generate SHAP explanation for XGBoost fundamental prediction.
        Returns feature importance dict for this specific stock.
        """
        try:
            import xgboost as xgb
            import shap
            import os
            import pickle
            
            if not os.path.exists(model_path):
                return {"error": "Model not found for SHAP"}
                
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            
            # Extract features in the correct order based on fundamental_features.json
            import json
            feature_path = model_path.replace('fundamental_xgb.pkl', 'fundamental_features.json')
            if os.path.exists(feature_path):
                with open(feature_path, "r") as f:
                    ordered_features = json.load(f)
            else:
                ordered_features = list(features.keys())
                
            x = np.array([[features.get(f, 0.0) for f in ordered_features]])
            
            # To handle categorical properly or map cleanly, we use TreeExplainer directly
            explainer = shap.TreeExplainer(model)
            
            values = explainer.shap_values(x)
            
            # Predict to get the class index
            pred_class = int(model.predict(x)[0])
            
            # Handle shap values structure based on explainer outputs
            if isinstance(values, list):
                shap_vals = values[pred_class][0]
            elif values.shape[-1] > 1 and len(values.shape) == 3: # (nsamples, nfeatures, nclasses)
                shap_vals = values[0, :, pred_class]
            else:
                shap_vals = values[0]
                
            base_val = explainer.expected_value[pred_class] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
            
            return {
                'feature_names': ordered_features,
                'shap_values': [float(val) for val in shap_vals],
                'base_value': float(base_val),
                'predicted_class': pred_class,
            }
        except Exception as e:
            import traceback
            return {"error": f"SHAP extraction failed: {str(e)}\n{traceback.format_exc()}"}

    def _explain_graph_summary(self, graph_output: dict) -> dict:
        """
        Build narrative from graph output without needing the raw model.
        """
        summary = graph_output.get('graph_view_3d', {}).get('summary', {})
        top_conn = summary.get('strongest_connection', {})
        if top_conn:
            narrative = f"{top_conn.get('ticker')} (correlation={top_conn.get('correlation', 0):.2f}) is the strongest market signal influencing this stock."
        else:
            narrative = "No significant graph influence detected."

        return {
            'top_influencing_stocks': summary.get('related_stocks_top10_by_ticker', []),
            'interpretation': narrative
        }
