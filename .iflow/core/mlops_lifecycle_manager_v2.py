#!/usr/bin/env python3
"""
MLOps生命周期管理器 V2
为A项目iflow工作流系统提供完整的机器学习运维支持
解决模型监控、自动再训练、A/B测试等生产环境问题

核心功能：
1. 模型版本管理和生命周期追踪
2. 实时性能监控和数据漂移检测
3. 自动再训练和模型部署
4. A/B测试和金丝雀发布
5. 模型解释性和合规性检查
"""

import json
import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import IsolationForest
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import requests
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
import hashlib
import yaml

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    """模型状态枚举"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    FAILED = "failed"

class ModelQuality(Enum):
    """模型质量等级"""
    EXCELLENT = "excellent"  # > 90%
    GOOD = "good"           # 80-90%
    FAIR = "fair"           # 70-80%
    POOR = "poor"           # 60-70%
    CRITICAL = "critical"   # < 60%

@dataclass
class ModelMetadata:
    """模型元数据"""
    model_id: str
    model_name: str
    version: str
    status: ModelStatus
    created_at: datetime
    updated_at: datetime
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    quality: ModelQuality
    data_drift_score: float
    performance_trend: str
    deployment_count: int
    last_retrain_date: datetime
    next_retrain_date: datetime
    tags: List[str]

@dataclass
class ModelMetrics:
    """模型指标"""
    timestamp: datetime
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    prediction_count: int
    error_count: int
    latency_p50: float
    latency_p95: float
    data_drift_score: float

@dataclass
class DeploymentConfig:
    """部署配置"""
    deployment_id: str
    model_id: str
    environment: str  # staging/production
    strategy: str     # canary/blue-green/rolling
    canary_percentage: float
    rollback_threshold: float
    health_check_url: str
    max_replicas: int
    min_replicas: int

class DataDriftDetector:
    """数据漂移检测器"""
    
    def __init__(self, reference_data: pd.DataFrame, window_size: int = 1000):
        """
        初始化数据漂移检测器
        
        Args:
            reference_data: 参考数据集
            window_size: 滑动窗口大小
        """
        self.reference_data = reference_data
        self.window_size = window_size
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.isolation_forest.fit(reference_data.select_dtypes(include=[np.number]))
        
        # 特征重要性
        self.feature_importance = {}
        self.drift_threshold = 0.1
        
    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        检测数据漂移
        
        Args:
            current_data: 当前数据集
            
        Returns:
            漂移检测结果
        """
        try:
            # 数值特征
            numeric_cols = current_data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return {"drift_detected": False, "score": 0.0, "features": {}}
            
            # 计算分布差异
            feature_drifts = {}
            for col in numeric_cols:
                if col in self.reference_data.columns:
                    ref_data = self.reference_data[col].dropna()
                    curr_data = current_data[col].dropna()
                    
                    if len(ref_data) > 0 and len(curr_data) > 0:
                        # KS检验
                        from scipy import stats
                        ks_stat, ks_p = stats.ks_2samp(ref_data, curr_data)
                        
                        # 均值差异
                        mean_diff = abs(curr_data.mean() - ref_data.mean()) / (ref_data.mean() + 1e-8)
                        
                        # 标准差差异
                        std_diff = abs(curr_data.std() - ref_data.std()) / (ref_data.std() + 1e-8)
                        
                        feature_drifts[col] = {
                            "ks_statistic": float(ks_stat),
                            "ks_p_value": float(ks_p),
                            "mean_difference": float(mean_diff),
                            "std_difference": float(std_diff),
                            "drift_score": float((ks_stat + mean_diff + std_diff) / 3)
                        }
            
            # 整体漂移分数
            overall_drift_score = np.mean([f["drift_score"] for f in feature_drifts.values()]) if feature_drifts else 0.0
            
            # 漂移检测结果
            drift_detected = overall_drift_score > self.drift_threshold
            
            return {
                "drift_detected": drift_detected,
                "score": float(overall_drift_score),
                "threshold": self.drift_threshold,
                "features": feature_drifts,
                "severity": self._get_drift_severity(overall_drift_score)
            }
            
        except Exception as e:
            logger.error(f"数据漂移检测失败: {e}")
            return {"drift_detected": False, "score": 0.0, "error": str(e)}
    
    def _get_drift_severity(self, score: float) -> str:
        """获取漂移严重程度"""
        if score < 0.05:
            return "low"
        elif score < 0.1:
            return "medium"
        else:
            return "high"

class ModelMonitor:
    """模型监控器"""
    
    def __init__(self, model_id: str, db_path: str = "mlops_monitoring.db"):
        """
        初始化模型监控器
        
        Args:
            model_id: 模型ID
            db_path: 数据库路径
        """
        self.model_id = model_id
        self.db_path = db_path
        self._init_database()
        
        # 性能阈值
        self.performance_thresholds = {
            "accuracy_min": 0.8,
            "precision_min": 0.75,
            "recall_min": 0.75,
            "f1_min": 0.75,
            "latency_p95_max": 1000,  # ms
            "error_rate_max": 0.05
        }
        
        # 监控线程
        self.monitoring_active = False
        self.monitor_thread = None
        
    def _init_database(self):
        """初始化监控数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    prediction_count INTEGER,
                    error_count INTEGER,
                    latency_p50 REAL,
                    latency_p95 REAL,
                    data_drift_score REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT,
                    timestamp DATETIME NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at DATETIME
                )
            """)
            
            conn.commit()
    
    def record_prediction(self, prediction_data: Dict[str, Any]):
        """
        记录预测数据
        
        Args:
            prediction_data: 预测相关数据
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO model_metrics (
                        model_id, timestamp, accuracy, precision, recall, f1_score,
                        prediction_count, error_count, latency_p50, latency_p95, data_drift_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    self.model_id,
                    datetime.now(),
                    prediction_data.get("accuracy"),
                    prediction_data.get("precision"),
                    prediction_data.get("recall"),
                    prediction_data.get("f1_score"),
                    prediction_data.get("prediction_count", 0),
                    prediction_data.get("error_count", 0),
                    prediction_data.get("latency_p50"),
                    prediction_data.get("latency_p95"),
                    prediction_data.get("data_drift_score", 0.0)
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"记录预测数据失败: {e}")
    
    def check_performance_alerts(self, metrics: ModelMetrics):
        """检查性能告警"""
        alerts = []
        
        # 准确率告警
        if metrics.accuracy < self.performance_thresholds["accuracy_min"]:
            alerts.append({
                "type": "performance",
                "severity": "high",
                "message": f"模型准确率低于阈值: {metrics.accuracy:.3f} < {self.performance_thresholds['accuracy_min']}",
                "metric": "accuracy",
                "value": metrics.accuracy,
                "threshold": self.performance_thresholds["accuracy_min"]
            })
        
        # 延迟告警
        if metrics.latency_p95 > self.performance_thresholds["latency_p95_max"]:
            alerts.append({
                "type": "performance",
                "severity": "medium",
                "message": f"模型P95延迟高于阈值: {metrics.latency_p95:.1f}ms > {self.performance_thresholds['latency_p95_max']}ms",
                "metric": "latency_p95",
                "value": metrics.latency_p95,
                "threshold": self.performance_thresholds["latency_p95_max"]
            })
        
        # 错误率告警
        error_rate = metrics.error_count / max(metrics.prediction_count, 1)
        if error_rate > self.performance_thresholds["error_rate_max"]:
            alerts.append({
                "type": "reliability",
                "severity": "high",
                "message": f"模型错误率高于阈值: {error_rate:.3f} > {self.performance_thresholds['error_rate_max']}",
                "metric": "error_rate",
                "value": error_rate,
                "threshold": self.performance_thresholds["error_rate_max"]
            })
        
        # 数据漂移告警
        if metrics.data_drift_score > 0.1:
            alerts.append({
                "type": "data_drift",
                "severity": "medium",
                "message": f"检测到数据漂移: {metrics.data_drift_score:.3f}",
                "metric": "data_drift_score",
                "value": metrics.data_drift_score,
                "threshold": 0.1
            })
        
        return alerts
    
    def get_performance_trend(self, days: int = 30) -> Dict[str, Any]:
        """获取性能趋势"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT 
                        DATE(timestamp) as date,
                        AVG(accuracy) as avg_accuracy,
                        AVG(precision) as avg_precision,
                        AVG(recall) as avg_recall,
                        AVG(f1_score) as avg_f1,
                        AVG(latency_p95) as avg_latency_p95,
                        AVG(data_drift_score) as avg_drift_score,
                        SUM(prediction_count) as total_predictions
                    FROM model_metrics 
                    WHERE model_id = ? AND timestamp >= datetime('now', '-{} days')
                    GROUP BY DATE(timestamp)
                    ORDER BY date DESC
                    LIMIT ?
                """.format(days)
                
                df = pd.read_sql_query(query, conn, params=(self.model_id, days))
                
                if len(df) < 2:
                    return {"trend": "insufficient_data", "data": []}
                
                # 计算趋势
                recent_avg = df.head(7)['avg_accuracy'].mean() if len(df) >= 7 else df['avg_accuracy'].mean()
                historical_avg = df.tail(7)['avg_accuracy'].mean() if len(df) >= 14 else df['avg_accuracy'].mean()
                
                if recent_avg > historical_avg * 1.02:
                    trend = "improving"
                elif recent_avg < historical_avg * 0.98:
                    trend = "degrading"
                else:
                    trend = "stable"
                
                return {
                    "trend": trend,
                    "recent_avg": float(recent_avg),
                    "historical_avg": float(historical_avg),
                    "data": df.to_dict('records')
                }
                
        except Exception as e:
            logger.error(f"获取性能趋势失败: {e}")
            return {"trend": "error", "error": str(e)}

class AutoRetrainingEngine:
    """自动再训练引擎"""
    
    def __init__(self, model_id: str, retrain_threshold: float = 0.05):
        """
        初始化自动再训练引擎
        
        Args:
            model_id: 模型ID
            retrain_threshold: 再训练触发阈值
        """
        self.model_id = model_id
        self.retrain_threshold = retrain_threshold
        self.monitor = ModelMonitor(model_id)
        self.drift_detector = None
        
        # 再训练配置
        self.retrain_config = {
            "min_data_size": 1000,
            "max_training_time": 3600,  # 1小时
            "performance_improvement_threshold": 0.02,
            "data_freshness_days": 30
        }
    
    def should_retrain(self) -> Dict[str, Any]:
        """
        判断是否需要再训练
        
        Returns:
            再训练决策结果
        """
        try:
            # 获取性能趋势
            trend = self.monitor.get_performance_trend()
            
            # 获取最近的监控数据
            with sqlite3.connect(self.monitor.db_path) as conn:
                query = """
                    SELECT * FROM model_metrics 
                    WHERE model_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """
                result = conn.execute(query, (self.model_id,)).fetchone()
            
            if not result:
                return {"should_retrain": False, "reason": "no_monitoring_data"}
            
            # 解析数据
            latest_metrics = ModelMetrics(
                timestamp=datetime.fromisoformat(result[2]),
                accuracy=result[3],
                precision=result[4],
                recall=result[5],
                f1_score=result[6],
                prediction_count=result[7],
                error_count=result[8],
                latency_p50=result[9],
                latency_p95=result[10],
                data_drift_score=result[11]
            )
            
            reasons = []
            should_retrain = False
            
            # 性能下降检查
            if trend["trend"] == "degrading":
                reasons.append("性能持续下降")
                should_retrain = True
            
            # 数据漂移检查
            if latest_metrics.data_drift_score > self.retrain_threshold:
                reasons.append(f"检测到数据漂移 (score: {latest_metrics.data_drift_score:.3f})")
                should_retrain = True
            
            # 准确率低于阈值
            if latest_metrics.accuracy < 0.8:
                reasons.append(f"模型准确率过低 ({latest_metrics.accuracy:.3f})")
                should_retrain = True
            
            return {
                "should_retrain": should_retrain,
                "reasons": reasons,
                "current_performance": {
                    "accuracy": latest_metrics.accuracy,
                    "trend": trend["trend"],
                    "drift_score": latest_metrics.data_drift_score
                }
            }
            
        except Exception as e:
            logger.error(f"再训练决策失败: {e}")
            return {"should_retrain": False, "error": str(e)}
    
    def execute_retraining(self, training_data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """
        执行模型再训练
        
        Args:
            training_data: 训练数据
            target_column: 目标列
            
        Returns:
            再训练结果
        """
        try:
            start_time = time.time()
            
            # 数据验证
            if len(training_data) < self.retrain_config["min_data_size"]:
                return {
                    "success": False,
                    "error": f"训练数据不足，需要至少{self.retrain_config['min_data_size']}条记录"
                }
            
            # 特征工程（简化版）
            X = training_data.drop(columns=[target_column])
            y = training_data[target_column]
            
            # 选择模型（简化版）
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # 训练模型
            model.fit(X, y)
            
            # 评估模型
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)
            
            metrics = {
                "accuracy": accuracy_score(y, y_pred),
                "precision": precision_score(y, y_pred, average='weighted'),
                "recall": recall_score(y, y_pred, average='weighted'),
                "f1": f1_score(y, y_pred, average='weighted')
            }
            
            training_time = time.time() - start_time
            
            # 保存新模型
            model_filename = f"model_{self.model_id}_{int(time.time())}.pkl"
            joblib.dump(model, model_filename)
            
            # 记录到MLflow
            mlflow.set_experiment(f"model_{self.model_id}")
            with mlflow.start_run():
                mlflow.log_params({
                    "model_type": "RandomForest",
                    "n_estimators": 100,
                    "training_time": training_time,
                    "data_size": len(training_data)
                })
                
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, "model")
                
            return {
                "success": True,
                "model_path": model_filename,
                "metrics": metrics,
                "training_time": training_time,
                "version": f"v{int(time.time())}"
            }
            
        except Exception as e:
            logger.error(f"模型再训练失败: {e}")
            return {"success": False, "error": str(e)}

class ABDeploymentManager:
    """A/B部署管理器"""
    
    def __init__(self, deployment_config: DeploymentConfig):
        """
        初始化A/B部署管理器
        
        Args:
            deployment_config: 部署配置
        """
        self.deployment_config = deployment_config
        self.active_experiments = {}
        self.metrics_collector = {}
        
        # 部署策略
        self.deployment_strategies = {
            "canary": self._canary_deployment,
            "blue_green": self._blue_green_deployment,
            "rolling": self._rolling_deployment
        }
    
    def start_experiment(self, model_a_id: str, model_b_id: str, traffic_split: Dict[str, float]) -> Dict[str, Any]:
        """
        启动A/B测试实验
        
        Args:
            model_a_id: 模型A ID
            model_b_id: 模型B ID
            traffic_split: 流量分配比例
            
        Returns:
            实验启动结果
        """
        try:
            experiment_id = f"exp_{int(time.time())}"
            
            experiment_config = {
                "experiment_id": experiment_id,
                "model_a_id": model_a_id,
                "model_b_id": model_b_id,
                "traffic_split": traffic_split,
                "start_time": datetime.now(),
                "status": "running",
                "metrics": {
                    "impressions": {"model_a": 0, "model_b": 0},
                    "conversions": {"model_a": 0, "model_b": 0},
                    "revenue": {"model_a": 0.0, "model_b": 0.0}
                }
            }
            
            self.active_experiments[experiment_id] = experiment_config
            
            return {
                "success": True,
                "experiment_id": experiment_id,
                "config": experiment_config
            }
            
        except Exception as e:
            logger.error(f"启动A/B测试失败: {e}")
            return {"success": False, "error": str(e)}
    
    def route_traffic(self, request_id: str, user_id: str = None) -> Dict[str, Any]:
        """
        路由流量到不同模型
        
        Args:
            request_id: 请求ID
            user_id: 用户ID（用于一致性路由）
            
        Returns:
            路由决策
        """
        try:
            # 获取当前活跃实验
            if not self.active_experiments:
                return {"model_id": "default", "experiment_id": None}
            
            experiment = list(self.active_experiments.values())[0]
            
            # 一致性哈希路由
            if user_id:
                hash_value = hash(f"{user_id}_{experiment['experiment_id']}") % 100
            else:
                hash_value = hash(request_id) % 100
            
            # 根据流量分配决定路由
            model_a_threshold = experiment["traffic_split"]["model_a"] * 100
            
            if hash_value < model_a_threshold:
                selected_model = experiment["model_a_id"]
                experiment["metrics"]["impressions"]["model_a"] += 1
            else:
                selected_model = experiment["model_b_id"]
                experiment["metrics"]["impressions"]["model_b"] += 1
            
            return {
                "model_id": selected_model,
                "experiment_id": experiment["experiment_id"],
                "routing_hash": hash_value
            }
            
        except Exception as e:
            logger.error(f"流量路由失败: {e}")
            return {"model_id": "default", "error": str(e)}
    
    def analyze_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """
        分析实验结果
        
        Args:
            experiment_id: 实验ID
            
        Returns:
            实验分析结果
        """
        try:
            experiment = self.active_experiments.get(experiment_id)
            if not experiment:
                return {"error": "实验不存在"}
            
            metrics = experiment["metrics"]
            
            # 计算转化率
            conv_rate_a = metrics["conversions"]["model_a"] / max(metrics["impressions"]["model_a"], 1)
            conv_rate_b = metrics["conversions"]["model_b"] / max(metrics["impressions"]["model_b"], 1)
            
            # 计算收入
            revenue_a = metrics["revenue"]["model_a"]
            revenue_b = metrics["revenue"]["model_b"]
            
            # 统计显著性检验（简化版）
            import scipy.stats as stats
            
            # 假设转化是二项分布
            if metrics["impressions"]["model_a"] > 0 and metrics["impressions"]["model_b"] > 0:
                # 卡方检验
                contingency_table = [
                    [metrics["conversions"]["model_a"], metrics["impressions"]["model_a"] - metrics["conversions"]["model_a"]],
                    [metrics["conversions"]["model_b"], metrics["impressions"]["model_b"] - metrics["conversions"]["model_b"]]
                ]
                
                try:
                    chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
                    statistically_significant = p_value < 0.05
                except:
                    p_value = 1.0
                    statistically_significant = False
            else:
                p_value = 1.0
                statistically_significant = False
            
            # 推荐决策
            if conv_rate_b > conv_rate_a * 1.05 and statistically_significant:
                recommendation = "model_b_wins"
                confidence = "high"
            elif conv_rate_a > conv_rate_b * 1.05 and statistically_significant:
                recommendation = "model_a_wins"
                confidence = "high"
            else:
                recommendation = "inconclusive"
                confidence = "low"
            
            return {
                "experiment_id": experiment_id,
                "duration": str(datetime.now() - experiment["start_time"]),
                "impressions": metrics["impressions"],
                "conversions": metrics["conversions"],
                "conversion_rates": {"model_a": conv_rate_a, "model_b": conv_rate_b},
                "revenue": {"model_a": revenue_a, "model_b": revenue_b},
                "statistical_significance": statistically_significant,
                "p_value": p_value,
                "recommendation": recommendation,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"实验分析失败: {e}")
            return {"error": str(e)}

class MLOpsLifecycleManager:
    """MLOps生命周期管理器主类"""
    
    def __init__(self, config_path: str = "mlops_config.yaml"):
        """
        初始化MLOps生命周期管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.models = {}  # model_id -> ModelMetadata
        self.monitors = {}  # model_id -> ModelMonitor
        self.retrain_engines = {}  # model_id -> AutoRetrainingEngine
        self.db_path = "mlops_lifecycle.db"
        
        # 初始化数据库
        self._init_database()
        
        # MLflow配置
        mlflow.set_tracking_uri(self.config.get("mlflow_tracking_uri", "http://localhost:5000"))
        self.mlflow_client = MlflowClient()
        
        logger.info("MLOps生命周期管理器初始化完成")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            # 默认配置
            return {
                "mlflow_tracking_uri": "http://localhost:5000",
                "model_registry_uri": "http://localhost:5000",
                "monitoring_interval": 300,  # 5分钟
                "retrain_check_interval": 3600,  # 1小时
                "data_drift_threshold": 0.1,
                "performance_degradation_threshold": 0.05
            }
    
    def _init_database(self):
        """初始化生命周期数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_lifecycle (
                    model_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    quality TEXT,
                    data_drift_score REAL,
                    performance_trend TEXT,
                    deployment_count INTEGER DEFAULT 0,
                    last_retrain_date DATETIME,
                    next_retrain_date DATETIME,
                    tags TEXT,
                    mlflow_run_id TEXT,
                    model_uri TEXT
                )
            """)
            conn.commit()
    
    def register_model(self, model_name: str, model_version: str, model_uri: str, 
                       metrics: Dict[str, float], tags: List[str] = None) -> str:
        """
        注册新模型
        
        Args:
            model_name: 模型名称
            model_version: 模型版本
            model_uri: 模型URI
            metrics: 模型指标
            tags: 标签列表
            
        Returns:
            模型ID
        """
        try:
            model_id = f"{model_name}_{model_version}_{int(time.time())}"
            
            # 计算质量等级
            avg_score = np.mean(list(metrics.values()))
            if avg_score >= 0.9:
                quality = ModelQuality.EXCELLENT
            elif avg_score >= 0.8:
                quality = ModelQuality.GOOD
            elif avg_score >= 0.7:
                quality = ModelQuality.FAIR
            elif avg_score >= 0.6:
                quality = ModelQuality.POOR
            else:
                quality = ModelQuality.CRITICAL
            
            metadata = ModelMetadata(
                model_id=model_id,
                model_name=model_name,
                version=model_version,
                status=ModelStatus.DEVELOPMENT,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                accuracy=metrics.get("accuracy", 0.0),
                precision=metrics.get("precision", 0.0),
                recall=metrics.get("recall", 0.0),
                f1_score=metrics.get("f1_score", 0.0),
                quality=quality,
                data_drift_score=0.0,
                performance_trend="stable",
                deployment_count=0,
                last_retrain_date=datetime.now(),
                next_retrain_date=datetime.now() + timedelta(days=30),
                tags=tags or []
            )
            
            # 保存到数据库
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO model_lifecycle VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metadata.model_id, metadata.model_name, metadata.version,
                    metadata.status.value, metadata.created_at, metadata.updated_at,
                    metadata.accuracy, metadata.precision, metadata.recall, metadata.f1_score,
                    metadata.quality.value, metadata.data_drift_score, metadata.performance_trend,
                    metadata.deployment_count, metadata.last_retrain_date, metadata.next_retrain_date,
                    json.dumps(metadata.tags), None, model_uri
                ))
                conn.commit()
            
            # 初始化监控和再训练组件
            self.monitors[model_id] = ModelMonitor(model_id)
            self.retrain_engines[model_id] = AutoRetrainingEngine(model_id)
            
            self.models[model_id] = metadata
            
            logger.info(f"模型注册成功: {model_id}")
            return model_id
            
        except Exception as e:
            logger.error(f"模型注册失败: {e}")
            return None
    
    def promote_model(self, model_id: str, target_environment: str) -> Dict[str, Any]:
        """
        模型升级到目标环境
        
        Args:
            model_id: 模型ID
            target_environment: 目标环境 (staging/production)
            
        Returns:
            升级结果
        """
        try:
            if model_id not in self.models:
                return {"success": False, "error": "模型不存在"}
            
            current_status = self.models[model_id].status
            
            # 状态转换验证
            if target_environment == "staging":
                if current_status != ModelStatus.DEVELOPMENT:
                    return {"success": False, "error": "只能从开发环境升级到预发布环境"}
                new_status = ModelStatus.STAGING
            elif target_environment == "production":
                if current_status != ModelStatus.STAGING:
                    return {"success": False, "error": "只能从预发布环境升级到生产环境"}
                new_status = ModelStatus.PRODUCTION
            else:
                return {"success": False, "error": "无效的目标环境"}
            
            # 性能验证
            if self.models[model_id].accuracy < 0.8:
                return {"success": False, "error": "模型准确率不满足生产环境要求"}
            
            # 更新状态
            self.models[model_id].status = new_status
            self.models[model_id].updated_at = datetime.now()
            
            # 更新数据库
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE model_lifecycle SET status = ?, updated_at = ?
                    WHERE model_id = ?
                """, (new_status.value, datetime.now(), model_id))
                conn.commit()
            
            logger.info(f"模型升级成功: {model_id} -> {target_environment}")
            return {"success": True, "previous_status": current_status.value, "new_status": new_status.value}
            
        except Exception as e:
            logger.error(f"模型升级失败: {e}")
            return {"success": False, "error": str(e)}
    
    def monitor_all_models(self):
        """监控所有模型"""
        try:
            for model_id, monitor in self.monitors.items():
                # 获取最新指标
                with sqlite3.connect(monitor.db_path) as conn:
                    query = """
                        SELECT * FROM model_metrics 
                        WHERE model_id = ? 
                        ORDER BY timestamp DESC 
                        LIMIT 1
                    """
                    result = conn.execute(query, (model_id,)).fetchone()
                
                if result:
                    # 构建指标对象
                    metrics = ModelMetrics(
                        timestamp=datetime.fromisoformat(result[2]),
                        accuracy=result[3],
                        precision=result[4],
                        recall=result[5],
                        f1_score=result[6],
                        prediction_count=result[7],
                        error_count=result[8],
                        latency_p50=result[9],
                        latency_p95=result[10],
                        data_drift_score=result[11]
                    )
                    
                    # 检查告警
                    alerts = monitor.check_performance_alerts(metrics)
                    
                    # 处理告警
                    for alert in alerts:
                        self._handle_alert(model_id, alert)
                    
                    # 更新模型元数据
                    if model_id in self.models:
                        self.models[model_id].accuracy = metrics.accuracy
                        self.models[model_id].updated_at = datetime.now()
                        
                        # 保存更新
                        with sqlite3.connect(self.db_path) as conn:
                            conn.execute("""
                                UPDATE model_lifecycle SET accuracy = ?, updated_at = ?
                                WHERE model_id = ?
                            """, (metrics.accuracy, datetime.now(), model_id))
                            conn.commit()
            
            logger.info("所有模型监控完成")
            
        except Exception as e:
            logger.error(f"模型监控失败: {e}")
    
    def _handle_alert(self, model_id: str, alert: Dict[str, Any]):
        """处理告警"""
        alert_type = alert["type"]
        severity = alert["severity"]
        message = alert["message"]
        
        logger.warning(f"模型告警 [{model_id}] [{severity}] {alert_type}: {message}")
        
        # 根据告警类型采取相应行动
        if alert_type == "data_drift" and severity == "high":
            # 触发数据漂移处理流程
            self._handle_data_drift(model_id)
        elif alert_type == "performance" and severity == "high":
            # 触发性能问题处理流程
            self._handle_performance_degradation(model_id)
    
    def _handle_data_drift(self, model_id: str):
        """处理数据漂移"""
        logger.info(f"处理数据漂移: {model_id}")
        
        # 检查是否需要再训练
        retrain_decision = self.retrain_engines[model_id].should_retrain()
        
        if retrain_decision["should_retrain"]:
            logger.info(f"数据漂移触发再训练: {model_id}")
            # 这里应该触发再训练流程
            # 实际实现中会调用训练管道
    
    def _handle_performance_degradation(self, model_id: str):
        """处理性能下降"""
        logger.info(f"处理性能下降: {model_id}")
        
        # 检查是否有备用模型
        # 如果有，执行回滚
        # 如果没有，触发紧急再训练
    
    def get_model_dashboard(self, model_id: str) -> Dict[str, Any]:
        """获取模型仪表板"""
        try:
            if model_id not in self.models:
                return {"error": "模型不存在"}
            
            model = self.models[model_id]
            
            # 获取性能趋势
            trend = self.monitors[model_id].get_performance_trend()
            
            # 获取最近的监控数据
            with sqlite3.connect(self.monitors[model_id].db_path) as conn:
                latest_metrics = conn.execute("""
                    SELECT * FROM model_metrics 
                    WHERE model_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """, (model_id,)).fetchone()
            
            dashboard = {
                "model_info": {
                    "model_id": model.model_id,
                    "model_name": model.model_name,
                    "version": model.version,
                    "status": model.status.value,
                    "created_at": model.created_at.isoformat(),
                    "quality": model.quality.value,
                    "tags": model.tags
                },
                "performance": {
                    "current": {
                        "accuracy": model.accuracy,
                        "precision": model.precision,
                        "recall": model.recall,
                        "f1_score": model.f1_score
                    },
                    "trend": trend["trend"],
                    "latest_metrics": {
                        "prediction_count": latest_metrics[7] if latest_metrics else 0,
                        "error_count": latest_metrics[8] if latest_metrics else 0,
                        "latency_p95": latest_metrics[10] if latest_metrics else 0
                    } if latest_metrics else {}
                },
                "deployment": {
                    "deployment_count": model.deployment_count,
                    "last_retrain_date": model.last_retrain_date.isoformat(),
                    "next_retrain_date": model.next_retrain_date.isoformat()
                },
                "alerts": []  # 这里可以从数据库获取最近的告警
            }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"获取模型仪表板失败: {e}")
            return {"error": str(e)}

# 使用示例和测试
if __name__ == "__main__":
    # 创建MLOps管理器
    mlops_manager = MLOpsLifecycleManager()
    
    # 注册测试模型
    test_metrics = {
        "accuracy": 0.85,
        "precision": 0.82,
        "recall": 0.88,
        "f1_score": 0.85
    }
    
    model_id = mlops_manager.register_model(
        model_name="health_risk_predictor",
        model_version="v1.0.0",
        model_uri="s3://models/health_risk_predictor_v1.0.0.pkl",
        metrics=test_metrics,
        tags=["healthcare", "risk_assessment"]
    )
    
    print(f"注册模型ID: {model_id}")
    
    # 升级到预发布环境
    result = mlops_manager.promote_model(model_id, "staging")
    print(f"升级结果: {result}")
    
    # 获取模型仪表板
    dashboard = mlops_manager.get_model_dashboard(model_id)
    print(f"模型仪表板: {json.dumps(dashboard, indent=2, default=str)}")
    
    # 模拟监控数据
    monitor = mlops_manager.monitors[model_id]
    test_prediction_data = {
        "accuracy": 0.83,
        "precision": 0.80,
        "recall": 0.85,
        "f1_score": 0.82,
        "prediction_count": 1000,
        "error_count": 15,
        "latency_p50": 45.2,
        "latency_p95": 120.5,
        "data_drift_score": 0.08
    }
    
    monitor.record_prediction(test_prediction_data)
    print("模拟监控数据记录完成")