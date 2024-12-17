# 贡献指南

## 开发流程

1. Fork项目并克隆到本地
```bash
git clone https://github.com/your-username/carla-test-platform.git
cd carla-test-platform
```

2. 创建新分支
```bash
git checkout -b feature/your-feature-name
```

3. 开发新功能
- 遵循代码风格指南
- 添加单元测试
- 更新文档

4. 提交代码
```bash
git add .
git commit -m "feat: add your feature description"
```

5. 推送到远程
```bash
git push origin feature/your-feature-name
```

6. 创建Pull Request

## 代码风格

- 使用Black格式化代码
- 遵循PEP 8规范
- 使用类型注解
- 编写文档字符串

## 提交规范

使用Angular提交规范:

- feat: 新功能
- fix: 修复bug
- docs: 文档更新
- style: 代码风格修改
- refactor: 重构代码
- test: 添加测试
- chore: 构建过程或辅助工具的变动

## 分支管理

- main: 主分支,保持稳定
- develop: 开发分支
- feature/*: 功能分支
- bugfix/*: 修复分支
- release/*: 发布分支

## 代码审查

- 所有代码必须经过审查才能合并
- 至少需要一个审查者批准
- CI检查必须通过

## 测试要求

- 单元测试覆盖率>80%
- 添加集成测试
- 性能测试不能退化 