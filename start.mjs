#!/usr/bin/env node
/**
 * Toolify 管理脚本
 * 支持启动、停止、重启、安装依赖等操作
 */

import { spawn, execSync } from 'child_process'
import { createInterface } from 'readline'
import { existsSync, readFileSync, writeFileSync, unlinkSync } from 'fs'
import { join, dirname } from 'path'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

const PID_FILE = join(__dirname, '.toolify.pid')
const LOG_FILE = join(__dirname, 'toolify.log')
const CONFIG_FILE = join(__dirname, 'config.yaml')
const CONFIG_EXAMPLE = join(__dirname, 'config.example.yaml')

// ANSI 颜色
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
  gray: '\x1b[90m'
}

function log(msg, color = 'reset') {
  console.log(`${colors[color]}${msg}${colors.reset}`)
}

function logSuccess(msg) { log(`✅ ${msg}`, 'green') }
function logError(msg) { log(`❌ ${msg}`, 'red') }
function logInfo(msg) { log(`ℹ️  ${msg}`, 'cyan') }
function logWarn(msg) { log(`⚠️  ${msg}`, 'yellow') }

// 检查 Python 是否可用
function getPythonCommand() {
  const commands = ['python3', 'python']
  for (const cmd of commands) {
    try {
      execSync(`${cmd} --version`, { stdio: 'ignore' })
      return cmd
    } catch {
      continue
    }
  }
  return null
}

// 检查 pip 是否可用
function getPipCommand() {
  const commands = ['pip3', 'pip']
  for (const cmd of commands) {
    try {
      execSync(`${cmd} --version`, { stdio: 'ignore' })
      return cmd
    } catch {
      continue
    }
  }
  return null
}

// 获取运行中的进程 PID
function getRunningPid() {
  if (!existsSync(PID_FILE)) {
    return null
  }

  const pid = parseInt(readFileSync(PID_FILE, 'utf8').trim(), 10)

  // 检查进程是否存在
  try {
    process.kill(pid, 0)
    return pid
  } catch {
    // 进程不存在，清理 PID 文件
    unlinkSync(PID_FILE)
    return null
  }
}

// 检查服务状态
function checkStatus() {
  const pid = getRunningPid()
  if (pid) {
    logSuccess(`Toolify 正在运行 (PID: ${pid})`)
    return true
  } else {
    logInfo('Toolify 未运行')
    return false
  }
}

// 安装依赖
function installDependencies() {
  const pip = getPipCommand()
  if (!pip) {
    logError('找不到 pip，请先安装 Python')
    return false
  }

  logInfo('正在安装依赖...')

  try {
    execSync(`${pip} install -r requirements.txt`, {
      cwd: __dirname,
      stdio: 'inherit'
    })
    logSuccess('依赖安装完成')
    return true
  } catch (error) {
    logError(`依赖安装失败: ${error.message}`)
    return false
  }
}

// 检查配置文件
function checkConfig() {
  if (!existsSync(CONFIG_FILE)) {
    if (existsSync(CONFIG_EXAMPLE)) {
      logWarn('config.yaml 不存在，正在从 config.example.yaml 复制...')
      const content = readFileSync(CONFIG_EXAMPLE, 'utf8')
      writeFileSync(CONFIG_FILE, content)
      logSuccess('已创建 config.yaml，请根据需要修改配置')
      return true
    } else {
      logError('config.yaml 和 config.example.yaml 都不存在')
      return false
    }
  }
  return true
}

// 启动服务
function startService() {
  const pid = getRunningPid()
  if (pid) {
    logWarn(`Toolify 已在运行 (PID: ${pid})`)
    return false
  }

  const python = getPythonCommand()
  if (!python) {
    logError('找不到 Python，请先安装 Python 3')
    return false
  }

  if (!checkConfig()) {
    return false
  }

  logInfo('正在启动 Toolify...')

  const child = spawn(python, ['main.py'], {
    cwd: __dirname,
    detached: true,
    stdio: ['ignore', 'pipe', 'pipe']
  })

  // 写入 PID 文件
  writeFileSync(PID_FILE, child.pid.toString())

  // 日志输出
  const logStream = existsSync(LOG_FILE)
    ? require('fs').createWriteStream(LOG_FILE, { flags: 'a' })
    : require('fs').createWriteStream(LOG_FILE)

  child.stdout.on('data', (data) => {
    logStream.write(data)
  })

  child.stderr.on('data', (data) => {
    logStream.write(data)
  })

  child.unref()

  // 等待一下检查是否成功启动
  setTimeout(() => {
    if (getRunningPid()) {
      logSuccess(`Toolify 已启动 (PID: ${child.pid})`)
      logInfo(`日志文件: ${LOG_FILE}`)
    } else {
      logError('Toolify 启动失败，请检查日志')
    }
  }, 1000)

  return true
}

// 停止服务
function stopService() {
  const pid = getRunningPid()
  if (!pid) {
    logInfo('Toolify 未在运行')
    return false
  }

  logInfo(`正在停止 Toolify (PID: ${pid})...`)

  try {
    process.kill(pid, 'SIGTERM')

    // 等待进程结束
    let attempts = 0
    const checkInterval = setInterval(() => {
      attempts++
      try {
        process.kill(pid, 0)
        if (attempts > 10) {
          // 强制杀死
          process.kill(pid, 'SIGKILL')
          clearInterval(checkInterval)
          if (existsSync(PID_FILE)) unlinkSync(PID_FILE)
          logSuccess('Toolify 已强制停止')
        }
      } catch {
        clearInterval(checkInterval)
        if (existsSync(PID_FILE)) unlinkSync(PID_FILE)
        logSuccess('Toolify 已停止')
      }
    }, 500)

    return true
  } catch (error) {
    logError(`停止失败: ${error.message}`)
    if (existsSync(PID_FILE)) unlinkSync(PID_FILE)
    return false
  }
}

// 重启服务
async function restartService() {
  logInfo('正在重启 Toolify...')
  stopService()

  // 等待进程完全停止
  await new Promise(resolve => setTimeout(resolve, 2000))

  startService()
}

// 查看日志
function viewLogs(lines = 50) {
  if (!existsSync(LOG_FILE)) {
    logInfo('日志文件不存在')
    return
  }

  logInfo(`最近 ${lines} 行日志:`)
  console.log(colors.gray + '─'.repeat(60) + colors.reset)

  try {
    const content = readFileSync(LOG_FILE, 'utf8')
    const logLines = content.trim().split('\n')
    const lastLines = logLines.slice(-lines)
    console.log(lastLines.join('\n'))
  } catch (error) {
    logError(`读取日志失败: ${error.message}`)
  }

  console.log(colors.gray + '─'.repeat(60) + colors.reset)
}

// 清除日志
function clearLogs() {
  if (existsSync(LOG_FILE)) {
    unlinkSync(LOG_FILE)
    logSuccess('日志已清除')
  } else {
    logInfo('日志文件不存在')
  }
}

// 显示菜单
function showMenu() {
  console.log()
  log('╔════════════════════════════════════════╗', 'cyan')
  log('║        🛠️  Toolify 管理控制台          ║', 'cyan')
  log('╠════════════════════════════════════════╣', 'cyan')
  log('║  1. 启动服务                           ║', 'cyan')
  log('║  2. 停止服务                           ║', 'cyan')
  log('║  3. 重启服务                           ║', 'cyan')
  log('║  4. 查看状态                           ║', 'cyan')
  log('║  5. 安装依赖                           ║', 'cyan')
  log('║  6. 查看日志                           ║', 'cyan')
  log('║  7. 清除日志                           ║', 'cyan')
  log('║  0. 退出                               ║', 'cyan')
  log('╚════════════════════════════════════════╝', 'cyan')
  console.log()
}

// 命令行参数处理
function handleArgs() {
  const args = process.argv.slice(2)

  if (args.length === 0) {
    return false // 进入交互模式
  }

  const command = args[0].toLowerCase()

  switch (command) {
    case 'start':
      startService()
      break
    case 'stop':
      stopService()
      break
    case 'restart':
      restartService()
      break
    case 'status':
      checkStatus()
      break
    case 'install':
      installDependencies()
      break
    case 'logs':
      viewLogs(parseInt(args[1]) || 50)
      break
    case 'clear-logs':
      clearLogs()
      break
    case 'help':
    case '-h':
    case '--help':
      console.log(`
${colors.bright}Toolify 管理脚本${colors.reset}

${colors.cyan}用法:${colors.reset}
  node start.mjs [command]

${colors.cyan}命令:${colors.reset}
  start       启动服务
  stop        停止服务
  restart     重启服务
  status      查看状态
  install     安装依赖
  logs [n]    查看最近 n 行日志 (默认 50)
  clear-logs  清除日志
  help        显示帮助

${colors.cyan}示例:${colors.reset}
  node start.mjs start      # 启动服务
  node start.mjs logs 100   # 查看最近 100 行日志
`)
      break
    default:
      logError(`未知命令: ${command}`)
      logInfo('使用 node start.mjs help 查看帮助')
  }

  return true
}

// 交互式菜单
async function interactiveMenu() {
  const rl = createInterface({
    input: process.stdin,
    output: process.stdout
  })

  const question = (prompt) => new Promise(resolve => rl.question(prompt, resolve))

  while (true) {
    showMenu()
    const choice = await question(`${colors.bright}请选择操作 [0-7]: ${colors.reset}`)

    switch (choice.trim()) {
      case '1':
        startService()
        break
      case '2':
        stopService()
        break
      case '3':
        await restartService()
        break
      case '4':
        checkStatus()
        break
      case '5':
        installDependencies()
        break
      case '6':
        viewLogs()
        break
      case '7':
        clearLogs()
        break
      case '0':
      case 'q':
      case 'quit':
      case 'exit':
        log('\n👋 再见!', 'green')
        rl.close()
        process.exit(0)
      default:
        logWarn('无效选择，请输入 0-7')
    }

    await question(`\n${colors.gray}按 Enter 继续...${colors.reset}`)
  }
}

// 主函数
async function main() {
  console.log(`${colors.bright}${colors.blue}
  ╔╦╗┌─┐┌─┐┬  ┬┌─┐┬ ┬
   ║ │ ││ ││  │├┤ └┬┘
   ╩ └─┘└─┘┴─┘┴└   ┴
${colors.reset}`)
  log('  Function Calling Middleware for LLMs', 'gray')
  console.log()

  // 如果有命令行参数，执行命令后退出
  if (handleArgs()) {
    return
  }

  // 否则进入交互模式
  await interactiveMenu()
}

main().catch(console.error)
