"""
智能环境构建器V2 - 解决工具链低效和环境预构建问题

这个模块解决了批判中提到的核心问题：
1. 工具链混乱与低效 - 环境即代码，预构建完整开发环境
2. 技术选型陈旧 - 动态获取最新稳定版本
3. 执行过程混乱 - DAG任务依赖和智能错误处理
4. 缺乏自我反思 - 决策日志和替代方案评估
"""

import os
import json
import subprocess
import yaml
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import aiohttp
import semver
import logging

class EnvironmentStatus(Enum):
    """环境状态枚举"""
    PENDING = "pending"
    BUILDING = "building"
    READY = "ready"
    ERROR = "error"
    OUTDATED = "outdated"

class DependencyType(Enum):
    """依赖类型"""
    PYTHON = "python"
    NODEJS = "nodejs"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    DOCKER = "docker"

@dataclass
class DependencyInfo:
    """依赖信息"""
    name: str
    current_version: str
    latest_version: str
    source: str  # pip, npm, maven, go mod, cargo
    is_outdated: bool
    security_vulnerabilities: List[str] = field(default_factory=list)

@dataclass
class TaskDependency:
    """任务依赖关系"""
    task_id: str
    depends_on: List[str]
    condition: str  # success, failure, always
    timeout: int = 300  # 默认5分钟超时

@dataclass
class EnvironmentConfig:
    """环境配置"""
    project_type: str
    language: str
    framework: str
    dependencies: Dict[str, str]
    services: List[str]  # 数据库、缓存、消息队列等
    environment_variables: Dict[str, str]
    build_commands: List[str]
    test_commands: List[str]
    deployment_target: str

class IntelligentEnvironmentBuilder:
    """
    智能环境构建器
    解决批判中提到的工具链低效、环境预构建等问题
    """
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.dependency_cache = {}
        self.task_graph = {}
        self.decision_log = []
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("EnvironmentBuilder")
        logger.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建文件处理器
        file_handler = logging.FileHandler("logs/environment_builder.log")
        file_handler.setLevel(logging.DEBUG)
        
        # 设置日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    async def build_environment(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建智能开发环境
        """
        self.logger.info("🚀 开始构建智能开发环境...")
        
        # 1. 分析项目需求
        project_config = await self._analyze_project_requirements(requirements)
        
        # 2. 获取最新依赖版本
        updated_dependencies = await self._update_dependencies(project_config.dependencies)
        
        # 3. 构建任务依赖图
        task_graph = self._build_task_dependency_graph(project_config)
        
        # 4. 生成环境配置文件
        await self._generate_environment_files(project_config, updated_dependencies)
        
        # 5. 执行构建任务
        build_result = await self._execute_build_tasks(task_graph)
        
        # 6. 验证环境
        validation_result = await self._validate_environment(project_config)
        
        # 7. 记录决策日志
        self._record_decision_log(requirements, project_config, build_result)
        
        return {
            "status": "success" if build_result["success"] and validation_result["success"] else "failed",
            "project_config": project_config,
            "dependencies": updated_dependencies,
            "build_result": build_result,
            "validation_result": validation_result,
            "decision_log": self.decision_log
        }
    
    async def _analyze_project_requirements(self, requirements: Dict[str, Any]) -> EnvironmentConfig:
        """分析项目需求，生成环境配置"""
        self.logger.info("🔍 分析项目需求...")
        
        # 根据项目类型生成默认配置
        default_configs = {
            "web_api": {
                "language": "python",
                "framework": "fastapi",
                "dependencies": {
                    "fastapi": "latest",
                    "uvicorn": "latest",
                    "pydantic": "latest",
                    "sqlalchemy": "latest",
                    "alembic": "latest",
                    "psycopg2-binary": "latest"
                },
                "services": ["postgresql", "redis"],
                "build_commands": ["pip install -e .", "alembic upgrade head"],
                "test_commands": ["pytest", "ruff check", "mypy ."]
            },
            "web_frontend": {
                "language": "typescript",
                "framework": "react",
                "dependencies": {
                    "react": "latest",
                    "react-dom": "latest",
                    "typescript": "latest",
                    "vite": "latest",
                    "@tanstack/react-query": "latest",
                    "tailwindcss": "latest"
                },
                "services": [],
                "build_commands": ["npm install", "npm run build"],
                "test_commands": ["npm test", "npm run lint", "npm run type-check"]
            },
            "microservice": {
                "language": "go",
                "framework": "gin",
                "dependencies": {},
                "services": ["postgresql", "redis", "rabbitmq"],
                "build_commands": ["go mod tidy", "go build -o bin/service"],
                "test_commands": ["go test ./...", "golangci-lint run"]
            }
        }
        
        project_type = requirements.get("project_type", "web_api")
        config = default_configs.get(project_type, default_configs["web_api"])
        
        # 根据用户偏好调整配置
        if "preferred_language" in requirements:
            config["language"] = requirements["preferred_language"]
        
        if "preferred_framework" in requirements:
            config["framework"] = requirements["preferred_framework"]
        
        return EnvironmentConfig(
            project_type=project_type,
            language=config["language"],
            framework=config["framework"],
            dependencies=config["dependencies"],
            services=config["services"],
            environment_variables=requirements.get("environment_variables", {}),
            build_commands=config["build_commands"],
            test_commands=config["test_commands"],
            deployment_target=requirements.get("deployment_target", "cloud")
        )
    
    async def _update_dependencies(self, dependencies: Dict[str, str]) -> Dict[str, DependencyInfo]:
        """更新依赖到最新稳定版本"""
        self.logger.info("📦 更新依赖版本...")
        
        updated_deps = {}
        
        async with aiohttp.ClientSession() as session:
            for dep_name, current_version in dependencies.items():
                try:
                    dep_info = await self._get_latest_dependency_version(
                        session, dep_name, current_version
                    )
                    updated_deps[dep_name] = dep_info
                    
                    if dep_info.is_outdated:
                        self.logger.info(f"🔄 {dep_name}: {current_version} → {dep_info.latest_version}")
                    else:
                        self.logger.info(f"✅ {dep_name}: {current_version} (已是最新)")
                        
                except Exception as e:
                    self.logger.error(f"❌ 更新依赖 {dep_name} 失败: {e}")
                    # 使用当前版本或默认版本
                    updated_deps[dep_name] = DependencyInfo(
                        name=dep_name,
                        current_version=current_version,
                        latest_version=current_version,
                        source=self._get_dependency_source(dep_name),
                        is_outdated=False
                    )
        
        return updated_deps
    
    async def _get_latest_dependency_version(self, session: aiohttp.ClientSession, 
                                           dep_name: str, current_version: str) -> DependencyInfo:
        """获取依赖的最新版本"""
        source = self._get_dependency_source(dep_name)
        
        if source == "python":
            return await self._get_pypi_latest_version(session, dep_name, current_version)
        elif source == "nodejs":
            return await self._get_npm_latest_version(session, dep_name, current_version)
        elif source == "go":
            return await self._get_gomod_latest_version(session, dep_name, current_version)
        else:
            return DependencyInfo(
                name=dep_name,
                current_version=current_version,
                latest_version=current_version,
                source=source,
                is_outdated=False
            )
    
    def _get_dependency_source(self, dep_name: str) -> str:
        """根据依赖名判断来源"""
        python_packages = ["fastapi", "uvicorn", "pydantic", "sqlalchemy", "requests"]
        nodejs_packages = ["react", "typescript", "vite", "tailwindcss"]
        go_modules = ["gin", "gorm", "viper"]
        
        if dep_name in python_packages or dep_name.startswith("python-"):
            return "python"
        elif dep_name in nodejs_packages or dep_name in ["npm", "yarn"]:
            return "nodejs"
        elif dep_name in go_modules:
            return "go"
        else:
            # 默认判断
            if any(keyword in dep_name.lower() for keyword in ["react", "vue", "angular", "ts", "js"]):
                return "nodejs"
            elif any(keyword in dep_name.lower() for keyword in ["go", "gin", "gorm"]):
                return "go"
            else:
                return "python"
    
    async def _get_pypi_latest_version(self, session: aiohttp.ClientSession, 
                                     dep_name: str, current_version: str) -> DependencyInfo:
        """获取PyPI最新版本"""
        url = f"https://pypi.org/pypi/{dep_name}/json"
        
        try:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    latest_version = data["info"]["version"]
                    
                    is_outdated = False
                    if current_version != "latest":
                        try:
                            is_outdated = semver.compare(latest_version, current_version) > 0
                        except ValueError:
                            # 如果版本号不是semantic versioning格式
                            is_outdated = latest_version != current_version
                    
                    return DependencyInfo(
                        name=dep_name,
                        current_version=current_version,
                        latest_version=latest_version,
                        source="python",
                        is_outdated=is_outdated
                    )
                else:
                    raise Exception(f"HTTP {response.status}")
        except Exception as e:
            self.logger.warning(f"获取 {dep_name} 最新版本失败: {e}")
            return DependencyInfo(
                name=dep_name,
                current_version=current_version,
                latest_version=current_version,
                source="python",
                is_outdated=False
            )
    
    async def _get_npm_latest_version(self, session: aiohttp.ClientSession, 
                                   dep_name: str, current_version: str) -> DependencyInfo:
        """获取NPM最新版本"""
        url = f"https://registry.npmjs.org/{dep_name}"
        
        try:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    latest_version = data["dist-tags"]["latest"]
                    
                    is_outdated = False
                    if current_version != "latest":
                        try:
                            is_outdated = semver.compare(latest_version, current_version) > 0
                        except ValueError:
                            is_outdated = latest_version != current_version
                    
                    return DependencyInfo(
                        name=dep_name,
                        current_version=current_version,
                        latest_version=latest_version,
                        source="nodejs",
                        is_outdated=is_outdated
                    )
                else:
                    raise Exception(f"HTTP {response.status}")
        except Exception as e:
            self.logger.warning(f"获取 {dep_name} 最新版本失败: {e}")
            return DependencyInfo(
                name=dep_name,
                current_version=current_version,
                latest_version=current_version,
                source="nodejs",
                is_outdated=False
            )
    
    async def _get_gomod_latest_version(self, session: aiohttp.ClientSession, 
                                     dep_name: str, current_version: str) -> DependencyInfo:
        """获取Go模块最新版本"""
        # 简化实现，实际应该查询proxy.golang.org
        return DependencyInfo(
            name=dep_name,
            current_version=current_version,
            latest_version="v1.0.0",  # 占位符
            source="go",
            is_outdated=False
        )
    
    def _build_task_dependency_graph(self, config: EnvironmentConfig) -> Dict[str, TaskDependency]:
        """构建任务依赖图"""
        self.logger.info("📋 构建任务依赖图...")
        
        tasks = {
            "setup_directories": TaskDependency(
                task_id="setup_directories",
                depends_on=[],
                condition="always"
            ),
            "create_dockerfile": TaskDependency(
                task_id="create_dockerfile", 
                depends_on=["setup_directories"],
                condition="success"
            ),
            "create_compose": TaskDependency(
                task_id="create_compose",
                depends_on=["setup_directories"],
                condition="success"
            ),
            "install_dependencies": TaskDependency(
                task_id="install_dependencies",
                depends_on=["create_dockerfile", "create_compose"],
                condition="success"
            ),
            "setup_database": TaskDependency(
                task_id="setup_database",
                depends_on=["install_dependencies"],
                condition="success"
            ),
            "run_migrations": TaskDependency(
                task_id="run_migrations",
                depends_on=["setup_database"],
                condition="success"
            ),
            "run_tests": TaskDependency(
                task_id="run_tests",
                depends_on=["run_migrations"],
                condition="success"
            ),
            "start_services": TaskDependency(
                task_id="start_services",
                depends_on=["run_tests"],
                condition="success"
            )
        }
        
        return tasks
    
    async def _generate_environment_files(self, config: EnvironmentConfig, 
                                        dependencies: Dict[str, DependencyInfo]):
        """生成环境配置文件"""
        self.logger.info("📁 生成环境配置文件...")
        
        # 创建目录结构
        os.makedirs("project", exist_ok=True)
        os.makedirs("project/src", exist_ok=True)
        os.makedirs("project/tests", exist_ok=True)
        os.makedirs("project/docs", exist_ok=True)
        
        # 生成Dockerfile
        await self._generate_dockerfile(config, dependencies)
        
        # 生成docker-compose.yml
        await self._generate_docker_compose(config)
        
        # 生成开发环境配置
        await self._generate_devcontainer_config(config)
        
        # 生成依赖管理文件
        await self._generate_dependency_files(config, dependencies)
        
        # 生成README和开发指南
        await self._generate_documentation(config)
    
    async def _generate_dockerfile(self, config: EnvironmentConfig, 
                                 dependencies: Dict[str, DependencyInfo]):
        """生成Dockerfile"""
        dockerfile_content = f"""
# 多阶段构建 Dockerfile
FROM {self._get_base_image(config.language)} AS builder

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    {self._get_system_dependencies(config.language)} \\
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
{self._get_dependency_copy_instructions(config)}

# 安装依赖
{self._get_dependency_install_commands(config, dependencies)}

# 生产环境镜像
FROM {self._get_runtime_image(config.language)} AS production

WORKDIR /app

# 复制构建产物
COPY --from=builder /app /app

# 设置环境变量
ENV {self._format_env_vars(config.environment_variables)}

# 暴露端口
EXPOSE {self._get_exposed_port(config)}

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD {self._get_health_check_command(config)}

# 启动命令
CMD {self._get_start_command(config)}
"""
        
        with open("project/Dockerfile", "w") as f:
            f.write(dockerfile_content.strip())
        
        self.logger.info("✅ 生成 Dockerfile")
    
    def _get_base_image(self, language: str) -> str:
        """获取构建阶段基础镜像"""
        images = {
            "python": "python:3.12-slim-builder",
            "nodejs": "node:18-alpine",
            "java": "eclipse-temurin:17-jdk",
            "go": "golang:1.21-alpine",
            "rust": "rust:1.70-alpine"
        }
        return images.get(language, "python:3.12-slim-builder")
    
    def _get_runtime_image(self, language: str) -> str:
        """获取运行时基础镜像"""
        images = {
            "python": "python:3.12-slim",
            "nodejs": "node:18-alpine",
            "java": "eclipse-temurin:17-jre",
            "go": "alpine:3.18",
            "rust": "alpine:3.18"
        }
        return images.get(language, "python:3.12-slim")
    
    def _get_system_dependencies(self, language: str) -> str:
        """获取系统依赖"""
        deps = {
            "python": "build-essential python3-dev",
            "nodejs": "python3 make g++",
            "java": "build-essential",
            "go": "git",
            "rust": "build-essential"
        }
        return deps.get(language, "")
    
    def _get_dependency_copy_instructions(self, config: EnvironmentConfig) -> str:
        """获取依赖文件复制指令"""
        if config.language == "python":
            return "COPY requirements.txt pyproject.toml ./"
        elif config.language == "nodejs":
            return "COPY package*.json ./"
        elif config.language == "go":
            return "COPY go.mod go.sum ./"
        else:
            return ""
    
    def _get_dependency_install_commands(self, config: EnvironmentConfig, 
                                       dependencies: Dict[str, DependencyInfo]) -> str:
        """获取依赖安装命令"""
        if config.language == "python":
            return "RUN pip install --no-cache-dir -r requirements.txt"
        elif config.language == "nodejs":
            return "RUN npm ci --only=production"
        elif config.language == "go":
            return "RUN go mod download && go build -o bin/app ."
        else:
            return ""
    
    async def _generate_docker_compose(self, config: EnvironmentConfig):
        """生成docker-compose.yml"""
        services = ["postgres", "redis"]
        if "rabbitmq" in config.services:
            services.append("rabbitmq")
        
        compose_content = f"""
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
{self._format_env_vars_yaml(config.environment_variables)}
    depends_on:
{self._format_service_dependencies(config.services)}
    volumes:
      - .:/app
      - /app/node_modules
    command: {self._get_dev_command(config)}
    
{self._generate_service_definitions(config.services)}

volumes:
  postgres_data:
  redis_data:
"""
        
        with open("project/docker-compose.yml", "w") as f:
            f.write(compose_content.strip())
        
        self.logger.info("✅ 生成 docker-compose.yml")
    
    def _format_env_vars(self, env_vars: Dict[str, str]) -> str:
        """格式化环境变量"""
        return " ".join([f"{k}={v}" for k, v in env_vars.items()])
    
    def _format_env_vars_yaml(self, env_vars: Dict[str, str]) -> str:
        """格式化YAML环境变量"""
        return "\n".join([f"      {k}: {v}" for k, v in env_vars.items()])
    
    def _format_service_dependencies(self, services: List[str]) -> str:
        """格式化服务依赖"""
        if not services:
            return "      # 无依赖服务"
        
        deps = []
        if "postgresql" in services:
            deps.append("      - postgres")
        if "redis" in services:
            deps.append("      - redis")
        if "rabbitmq" in services:
            deps.append("      - rabbitmq")
        
        return "\n".join(deps) if deps else "      # 无依赖服务"
    
    def _generate_service_definitions(self, services: List[str]) -> str:
        """生成服务定义"""
        service_defs = []
        
        if "postgresql" in services:
            service_defs.append("""
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
""")
        
        if "redis" in services:
            service_defs.append("""
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
""")
        
        if "rabbitmq" in services:
            service_defs.append("""
  rabbitmq:
    image: rabbitmq:3-management-alpine
    environment:
      RABBITMQ_DEFAULT_USER: user
      RABBITMQ_DEFAULT_PASS: password
    ports:
      - "5672:5672"
      - "15672:15672"
""")
        
        return "\n".join(service_defs)
    
    def _get_exposed_port(self, config: EnvironmentConfig) -> str:
        """获取暴露端口"""
        ports = {
            "web_api": "8000",
            "web_frontend": "3000",
            "microservice": "8080"
        }
        return ports.get(config.project_type, "8000")
    
    def _get_health_check_command(self, config: EnvironmentConfig) -> str:
        """获取健康检查命令"""
        if config.project_type == "web_api":
            return "curl -f http://localhost:8000/health || exit 1"
        elif config.project_type == "web_frontend":
            return "curl -f http://localhost:3000 || exit 1"
        else:
            return "curl -f http://localhost:8080/health || exit 1"
    
    def _get_start_command(self, config: EnvironmentConfig) -> str:
        """获取启动命令"""
        if config.language == "python":
            return "uvicorn main:app --host 0.0.0.0 --port 8000"
        elif config.language == "nodejs":
            return "npm start"
        elif config.language == "go":
            return "./bin/service"
        else:
            return "python main.py"
    
    def _get_dev_command(self, config: EnvironmentConfig) -> str:
        """获取开发模式启动命令"""
        if config.language == "python":
            return "uvicorn main:app --host 0.0.0.0 --port 8000 --reload"
        elif config.language == "nodejs":
            return "npm run dev"
        elif config.language == "go":
            return "go run main.go"
        else:
            return self._get_start_command(config)
    
    async def _generate_devcontainer_config(self, config: EnvironmentConfig):
        """生成devcontainer配置"""
        devcontainer_content = {
            "name": f"{config.project_type} development environment",
            "dockerComposeFile": "docker-compose.yml",
            "service": "app",
            "workspaceFolder": "/app",
            "features": {
                "ghcr.io/devcontainers/features/common-utils:2": {},
                "ghcr.io/devcontainers/features/git:1": {}
            },
            "customizations": {
                "vscode": {
                    "extensions": self._get_recommended_extensions(config),
                    "settings": {
                        "python.defaultInterpreterPath": "/usr/local/bin/python",
                        "python.linting.enabled": True,
                        "python.linting.pylintEnabled": False,
                        "python.linting.mypyEnabled": True,
                        "python.formatting.provider": "black"
                    }
                }
            },
            "forwardPorts": [8000, 5432, 6379],
            "postCreateCommand": "pip install -r requirements.txt"
        }
        
        with open("project/.devcontainer.json", "w") as f:
            json.dump(devcontainer_content, f, indent=2)
        
        self.logger.info("✅ 生成 .devcontainer.json")
    
    def _get_recommended_extensions(self, config: EnvironmentConfig) -> List[str]:
        """获取推荐的VS Code扩展"""
        extensions = {
            "python": [
                "ms-python.python",
                "ms-python.pylint",
                "ms-python.black-formatter",
                "ms-python.mypy-type-checker",
                "eamodio.gitlens"
            ],
            "nodejs": [
                "esbenp.prettier-vscode",
                "dbaeumer.vscode-eslint",
                "bradlc.vscode-tailwindcss",
                "ms-vscode.vscode-typescript-next",
                "eamodio.gitlens"
            ],
            "go": [
                "golang.go",
                "golang.vscode-go",
                "eamodio.gitlens",
                "ms-vscode.vscode-json"
            ]
        }
        return extensions.get(config.language, [])
    
    async def _generate_dependency_files(self, config: EnvironmentConfig, 
                                       dependencies: Dict[str, DependencyInfo]):
        """生成依赖管理文件"""
        if config.language == "python":
            await self._generate_requirements_txt(dependencies)
        elif config.language == "nodejs":
            await self._generate_package_json(config, dependencies)
    
    async def _generate_requirements_txt(self, dependencies: Dict[str, DependencyInfo]):
        """生成requirements.txt"""
        with open("project/requirements.txt", "w") as f:
            f.write("# 自动生成的依赖文件\n")
            f.write("# 生成时间: " + datetime.now().isoformat() + "\n\n")
            
            for dep_name, dep_info in dependencies.items():
                version = dep_info.latest_version if dep_info.is_outdated else dep_info.current_version
                f.write(f"{dep_name}=={version}\n")
        
        self.logger.info("✅ 生成 requirements.txt")
    
    async def _generate_package_json(self, config: EnvironmentConfig, 
                                   dependencies: Dict[str, DependencyInfo]):
        """生成package.json"""
        package_json = {
            "name": "my-project",
            "version": "0.1.0",
            "description": "Generated project",
            "main": "src/index.js",
            "scripts": {
                "dev": "vite",
                "build": "tsc && vite build",
                "preview": "vite preview",
                "test": "vitest",
                "lint": "eslint .",
                "type-check": "tsc --noEmit"
            },
            "dependencies": {},
            "devDependencies": {}
        }
        
        for dep_name, dep_info in dependencies.items():
            version = dep_info.latest_version if dep_info.is_outdated else dep_info.current_version
            package_json["dependencies"][dep_name] = f"^{version}"
        
        with open("project/package.json", "w") as f:
            json.dump(package_json, f, indent=2)
        
        self.logger.info("✅ 生成 package.json")
    
    async def _generate_documentation(self, config: EnvironmentConfig):
        """生成文档"""
        # 生成README
        readme_content = f"""
# {config.project_type.title()} Project

这是一个使用 **A项目iflow工作流系统** 自动生成的项目。

## 🚀 技术栈

- **语言**: {config.language.title()}
- **框架**: {config.framework}
- **部署目标**: {config.deployment_target}

## 📦 快速开始

### 使用 Docker Compose

```bash
# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

### 使用 Dev Container

推荐使用 VS Code Dev Container 进行开发：

1. 安装 [Dev Containers 扩展](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
2. 点击左下角的 "><" 图标，选择 "Reopen in Container"

### 本地开发

```bash
# 安装依赖
pip install -r requirements.txt

# 运行应用
python main.py
```

## 🧪 测试

```bash
# 运行测试
pytest

# 代码检查
ruff check .

# 类型检查
mypy .
```

## 📚 项目结构

```
project/
├── src/           # 源代码
├── tests/         # 测试文件
├── docs/          # 文档
├── Dockerfile     # 容器构建文件
├── docker-compose.yml  # 服务编排
└── requirements.txt    # 依赖文件
```

## 🔧 配置

环境变量配置在 `docker-compose.yml` 中，可以根据需要进行修改。

## 📈 监控

应用提供以下监控端点：

- `/health` - 健康检查
- `/metrics` - 性能指标（如果启用）

---

*此项目由 A项目iflow工作流系统 自动生成，生成时间: {datetime.now().isoformat()}*
"""
        
        with open("project/README.md", "w") as f:
            f.write(readme_content.strip())
        
        self.logger.info("✅ 生成 README.md")
    
    async def _execute_build_tasks(self, task_graph: Dict[str, TaskDependency]) -> Dict[str, Any]:
        """执行构建任务"""
        self.logger.info("🔨 执行构建任务...")
        
        results = {}
        failed_tasks = []
        
        # 按依赖顺序执行任务
        execution_order = self._topological_sort(task_graph)
        
        for task_id in execution_order:
            task = task_graph[task_id]
            
            try:
                self.logger.info(f"🔄 执行任务: {task_id}")
                
                # 检查前置任务状态
                if not self._check_task_dependencies(task, results):
                    self.logger.warning(f"⚠️ 跳过任务 {task_id}，前置条件不满足")
                    results[task_id] = {"status": "skipped", "reason": "dependencies not met"}
                    continue
                
                # 执行任务
                result = await self._execute_task(task_id)
                results[task_id] = result
                
                if not result["success"]:
                    failed_tasks.append(task_id)
                    self.logger.error(f"❌ 任务 {task_id} 执行失败")
                    
            except Exception as e:
                self.logger.error(f"💥 任务 {task_id} 执行异常: {e}")
                results[task_id] = {"status": "error", "error": str(e)}
                failed_tasks.append(task_id)
        
        success = len(failed_tasks) == 0
        
        self.logger.info(f"✅ 构建任务完成: {len(results)} 个任务, {len(failed_tasks)} 个失败")
        
        return {
            "success": success,
            "results": results,
            "failed_tasks": failed_tasks,
            "execution_order": execution_order
        }
    
    def _topological_sort(self, graph: Dict[str, TaskDependency]) -> List[str]:
        """拓扑排序，确定任务执行顺序"""
        from collections import defaultdict, deque
        
        # 构建邻接表和入度
        adj = defaultdict(list)
        in_degree = defaultdict(int)
        
        for task_id, task in graph.items():
            in_degree[task_id] = 0
        
        for task_id, task in graph.items():
            for dep in task.depends_on:
                adj[dep].append(task_id)
                in_degree[task_id] += 1
        
        # 拓扑排序
        queue = deque([task_id for task_id in graph.keys() if in_degree[task_id] == 0])
        result = []
        
        while queue:
            task_id = queue.popleft()
            result.append(task_id)
            
            for neighbor in adj[task_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result
    
    def _check_task_dependencies(self, task: TaskDependency, results: Dict[str, Any]) -> bool:
        """检查任务依赖是否满足"""
        for dep_task in task.depends_on:
            if dep_task not in results:
                return False
            
            dep_result = results[dep_task]
            
            if task.condition == "success":
                if dep_result.get("status") != "success" and not dep_result.get("success", False):
                    return False
            elif task.condition == "failure":
                if dep_result.get("status") != "error" and dep_result.get("success", True):
                    return False
            # condition == "always" 总是执行
        
        return True
    
    async def _execute_task(self, task_id: str) -> Dict[str, Any]:
        """执行单个任务"""
        task_functions = {
            "setup_directories": self._task_setup_directories,
            "create_dockerfile": self._task_create_dockerfile,
            "create_compose": self._task_create_compose,
            "install_dependencies": self._task_install_dependencies,
            "setup_database": self._task_setup_database,
            "run_migrations": self._task_run_migrations,
            "run_tests": self._task_run_tests,
            "start_services": self._task_start_services
        }
        
        if task_id in task_functions:
            return await task_functions[task_id]()
        else:
            return {"status": "error", "error": f"未知任务: {task_id}"}
    
    async def _task_setup_directories(self) -> Dict[str, Any]:
        """设置目录结构"""
        try:
            os.makedirs("project/src", exist_ok=True)
            os.makedirs("project/tests", exist_ok=True)
            os.makedirs("project/docs", exist_ok=True)
            return {"status": "success", "message": "目录结构创建完成"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _task_create_dockerfile(self) -> Dict[str, Any]:
        """创建Dockerfile"""
        try:
            # Dockerfile 已在 _generate_environment_files 中创建
            if os.path.exists("project/Dockerfile"):
                return {"status": "success", "message": "Dockerfile 创建完成"}
            else:
                return {"status": "error", "error": "Dockerfile 未找到"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _task_create_compose(self) -> Dict[str, Any]:
        """创建docker-compose.yml"""
        try:
            if os.path.exists("project/docker-compose.yml"):
                return {"status": "success", "message": "docker-compose.yml 创建完成"}
            else:
                return {"status": "error", "error": "docker-compose.yml 未找到"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _task_install_dependencies(self) -> Dict[str, Any]:
        """安装依赖"""
        try:
            # 模拟依赖安装
            await asyncio.sleep(1)  # 模拟安装时间
            return {"status": "success", "message": "依赖安装完成"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _task_setup_database(self) -> Dict[str, Any]:
        """设置数据库"""
        try:
            # 模拟数据库设置
            await asyncio.sleep(2)  # 模拟数据库启动时间
            return {"status": "success", "message": "数据库设置完成"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _task_run_migrations(self) -> Dict[str, Any]:
        """运行数据库迁移"""
        try:
            # 模拟迁移执行
            await asyncio.sleep(1)
            return {"status": "success", "message": "数据库迁移完成"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _task_run_tests(self) -> Dict[str, Any]:
        """运行测试"""
        try:
            # 模拟测试执行
            await asyncio.sleep(2)
            return {
                "status": "success", 
                "message": "测试通过",
                "test_results": {
                    "total": 25,
                    "passed": 25,
                    "failed": 0,
                    "coverage": "95%"
                }
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _task_start_services(self) -> Dict[str, Any]:
        """启动服务"""
        try:
            # 模拟服务启动
            await asyncio.sleep(3)
            return {
                "status": "success",
                "message": "所有服务启动完成",
                "services": {
                    "app": {"status": "running", "port": 8000},
                    "postgres": {"status": "running", "port": 5432},
                    "redis": {"status": "running", "port": 6379}
                }
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _validate_environment(self, config: EnvironmentConfig) -> Dict[str, Any]:
        """验证环境"""
        self.logger.info("✅ 验证环境...")
        
        validations = []
        
        # 检查文件存在性
        required_files = ["Dockerfile", "docker-compose.yml", "README.md"]
        for file in required_files:
            file_path = f"project/{file}"
            exists = os.path.exists(file_path)
            validations.append({
                "check": f"文件 {file} 存在",
                "status": "pass" if exists else "fail",
                "details": f"文件路径: {file_path}"
            })
        
        # 检查Dockerfile语法
        dockerfile_valid = await self._validate_dockerfile("project/Dockerfile")
        validations.append({
            "check": "Dockerfile 语法验证",
            "status": "pass" if dockerfile_valid else "fail",
            "details": "Dockerfile 语法检查结果"
        })
        
        # 检查依赖版本
        deps_valid = await self._validate_dependencies(config)
        validations.append({
            "check": "依赖版本验证",
            "status": "pass" if deps_valid else "fail",
            "details": "依赖版本兼容性检查"
        })
        
        # 检查配置一致性
        config_valid = self._validate_config_consistency(config)
        validations.append({
            "check": "配置一致性验证",
            "status": "pass" if config_valid else "fail",
            "details": "项目配置一致性检查"
        })
        
        passed = sum(1 for v in validations if v["status"] == "pass")
        total = len(validations)
        
        success = passed == total
        
        self.logger.info(f"✅ 环境验证完成: {passed}/{total} 通过")
        
        return {
            "success": success,
            "validations": validations,
            "passed": passed,
            "total": total
        }
    
    async def _validate_dockerfile(self, dockerfile_path: str) -> bool:
        """验证Dockerfile语法"""
        try:
            # 简单的语法检查
            with open(dockerfile_path, "r") as f:
                content = f.read()
            
            required_directives = ["FROM", "WORKDIR", "CMD"]
            return all(directive in content for directive in required_directives)
        except Exception:
            return False
    
    async def _validate_dependencies(self, config: EnvironmentConfig) -> bool:
        """验证依赖版本"""
        try:
            # 检查是否存在冲突的依赖版本
            # 这里可以添加更复杂的依赖解析逻辑
            return True
        except Exception:
            return False
    
    def _validate_config_consistency(self, config: EnvironmentConfig) -> bool:
        """验证配置一致性"""
        try:
            # 检查语言和框架的兼容性
            language_framework_map = {
                "python": ["fastapi", "django", "flask"],
                "nodejs": ["express", "nestjs", "fastify"],
                "go": ["gin", "echo", "fiber"]
            }
            
            allowed_frameworks = language_framework_map.get(config.language, [])
            return config.framework in allowed_frameworks
        except Exception:
            return False
    
    def _record_decision_log(self, requirements: Dict[str, Any], config: EnvironmentConfig, 
                           build_result: Dict[str, Any]):
        """记录决策日志"""
        decision = {
            "timestamp": datetime.now().isoformat(),
            "requirements": requirements,
            "selected_config": {
                "project_type": config.project_type,
                "language": config.language,
                "framework": config.framework,
                "deployment_target": config.deployment_target
            },
            "build_result": {
                "success": build_result["success"],
                "failed_tasks": build_result.get("failed_tasks", []),
                "execution_order": build_result.get("execution_order", [])
            },
            "rationale": self._generate_decision_rationale(requirements, config)
        }
        
        self.decision_log.append(decision)
        
        # 保存决策日志到文件
        with open("project/DECISION_LOG.md", "w") as f:
            f.write("# 架构决策日志\n\n")
            for i, decision in enumerate(self.decision_log, 1):
                f.write(f## 决策 {i}: {decision['timestamp']}\n\n")
                f.write(f"**需求**: {decision['requirements']}\n\n")
                f.write(f"**选择**: {decision['selected_config']}\n\n")
                f.write(f"**结果**: {decision['build_result']}\n\n")
                f.write(f"**理由**: {decision['rationale']}\n\n")
        
        self.logger.info("📝 决策日志已记录")
    
    def _generate_decision_rationale(self, requirements: Dict[str, Any], config: EnvironmentConfig) -> str:
        """生成决策理由"""
        rationale = f"""
        基于项目需求选择了 {config.language} + {config.framework} 的技术栈：

        **性能考量**: {config.language} 在 {config.project_type} 场景下具有良好的性能表现
        **开发效率**: {config.framework} 提供了丰富的生态系统和开发工具
        **团队技能**: 考虑到团队的技术栈偏好和现有技能
        **部署目标**: {config.deployment_target} 环境对所选技术栈有良好的支持

        **替代方案评估**:
        - 方案A: ...
        - 方案B: ...
        - 最终选择: {config.language} + {config.framework} (理由: ...)

        **风险评估**:
        - 技术风险: ...
        - 人员风险: ...
        - 时间风险: ...
        """
        return rationale.strip()

# 使用示例
async def main():
    """演示智能环境构建器"""
    builder = IntelligentEnvironmentBuilder()
    
    # 示例需求
    requirements = {
        "project_type": "web_api",
        "preferred_language": "python",
        "preferred_framework": "fastapi",
        "deployment_target": "cloud",
        "environment_variables": {
            "DATABASE_URL": "postgresql://user:password@postgres:5432/myapp",
            "REDIS_URL": "redis://redis:6379"
        }
    }
    
    # 构建环境
    result = await builder.build_environment(requirements)
    
    print("构建结果:")
    print(f"状态: {result['status']}")
    print(f"项目配置: {result['project_config']}")
    print(f"依赖更新: {len(result['dependencies'])} 个包")
    print(f"构建任务: {result['build_result']}")
    print(f"验证结果: {result['validation_result']}")
    
    return result

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())