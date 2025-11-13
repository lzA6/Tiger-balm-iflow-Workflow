# iFlow 工具集索引

## 工具分类

### 核心工具 (core)

- **enhanced-universal-agent** (`core/enhanced-universal-agent.py`)

### 量子计算工具 (quantum)

- **enterprise-quantum-deployment** (`quantum/enterprise-quantum-deployment.py`)
- **quantum-cloud-platform** (`quantum/quantum-cloud-platform.py`)
- **quantum-code-generator** (`quantum/quantum-code-generator.py`)
- **quantum-graphical-editor** (`quantum/quantum-graphical-editor.py`)
- **quantum-performance-optimizer** (`quantum/quantum-performance-optimizer.py`)

### 测试工具 (testing)

- **comprehensive-test-suite** (`testing/comprehensive-test-suite.py`)
- **test-runner** (`testing/test-runner.py`)

### 部署工具 (deployment)


### 分析工具 (analysis)

- **tool-call-validator** (`analysis/tool-call-validator.py`)

### 优化工具 (optimization)

- **self-evolution-engine** (`optimization/self-evolution-engine.py`)

### 安全工具 (security)

- **quantum-security-scanner** (`security/quantum-security-scanner.py`)

### MCP服务器工具 (mcp)

- **universal-mcp-server** (`mcp/universal-mcp-server.py`)

### 外部工具 (external)


## 使用说明

### 运行工具
```bash
# 从.iflow目录运行
python tools/{category}/{tool_name}.py [参数]

# 或使用iFlow CLI
python bin/iflow.py tool run {tool_name}
```

### 工具开发规范
1. 所有工具必须位于对应的分类目录中
2. 工具名称使用小写字母和连字符
3. 每个工具都应该有完整的文档字符串
4. 工具应该支持命令行参数
5. 错误处理要完善，提供有意义的错误信息

### 贡献新工具
1. 在合适的分类目录中创建工具文件
2. 更新此索引文件
3. 添加测试用例
4. 更新文档

---
更新时间: 2025-11-12
