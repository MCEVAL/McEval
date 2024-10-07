# 环境配置以及运行



## JavaScript
### 环境
```bash
Humaneval-x 提供的docker已经搭建完成
```
### 运行

```bash
node test.js
```

## Python 
### 环境
```bash
Humaneval-x 提供的docker已经搭建完成
```
### 运行
```bash
python 1.py
```

## CPP
### 环境
```bash
Humaneval-x 提供的docker已经搭建完成
```
### 运行
```bash
g++ -o 1 1.cpp && ./1
```


## C
### 环境
```bash
Humaneval-x 提供的docker已经搭建完成
```
### 运行
```bash
gcc -o 1 1.cpp && ./1
```


## Java
### 环境
```bash
Humaneval-x 提供的docker已经搭建完成
```
### 运行
```bash
java -ea tmp/1.java 
```


## Rust
### 环境
```bash
# 需要安装cargo，其余配置humaneval-x已经提供（.lock/.toml）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```
### 运行
```bash
cargo test
```


## go
### 环境
```bash
Humaneval-x 提供的docker已经搭建完成，需要注意文件名格式应为*_test.go
```
### 运行
```bash
go test 1_test.go
```


## Fortran
### 环境
```bash
apt install gfortran
```
### 运行
```bash
gfortran -o 1 1.f95
./1
```


## Typescript
### 环境
```bash
# npm config set registry https://registry.npm.taobao.org
npm install -g typescript
```
### 运行
```bash
tsc 1.ts
node 1.js
```


## Kotlin  
### 环境
```bash
apt-get install unzip
apt-get intall zip
curl -s "https://get.sdkman.io" | bash    
source "/root/.sdkman/bin/sdkman-init.sh"
sdk install kotlin

```

### 运行
```bash
kotlinc Main.kt -include-runtime -d Main.jar
java -jar -ea Main.jar
```


### 运行 （脚本运行，不需要编译）
```bash
kotlinc -script main.kts
// 将代码中的assert 替换为check 
```


## C-sharp
### 环境
```bash
wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
sudo apt-get update
apt-get install -y dotnet-sdk-8.0
```
### 运行
```bash
dotnet new console -n MyConsoleApp
cd MyConsoleApp/
dotnet build
dotnet run
```


## Scala
### 环境
```bash
curl -fL https://github.com/coursier/coursier/releases/latest/download/cs-x86_64-pc-linux.gz | gzip -d > cs && chmod +x cs && ./cs setup
export PATH=$PATH:/root/.local/share/coursier/bin
```
### 运行
```bash
scala main.scala
```


## Php
### 环境
```bash
apt-get install php
修改php.ini 文件，使得assert 生效： https://stackoverflow.com/questions/40129963/assert-is-not-working-in-php-so-simple-what-am-i-doing-wrong
php --ini // 查看ini文件位置
```
### 运行
```bash
php main.php
```

## Dart
### 环境
```bash
https://www.cnblogs.com/007sx/p/14505162.html
```

### 运行
```bash
vim pubspec.yaml
填入下边内容
name: my_app
environment:
  sdk: '^3.2.0'
dependencies:
  collection: ^1.15.0

dart run --enable-asserts test.dart 
```

## Pascal
### 环境
```bash
apt-get install fpc
```
### 运行
```bash
fpc test.pas
./test
```

## Julia

### 环境
```bash
访问Julia官方下载页面：Julia Downloads。
tar -xvzf julia-1.9.4-linux-x86_64.tar.gz
export PATH="/root/julia-1.9.4/bin:$PATH
source ./bashrc
```
## 执行
```
julia test.jl
```

## Ruby

### 环境
```bash
apt install ruby-full
```

### 执行
```
ruby test.rb
```

## Coffeescript

### 环境
```bash
npm install -g coffee-script
```
### 执行
```
coffee test.coffee
```
## Lua
### 环境
```bash
wget -c http://www.lua.org/ftp/lua-5.4.6.tar.gz
tar zxf lua-5.4.6.tar.gz
cd lua-5.4.6
make linux test
make install
```
### 执行
```
lua test.lua
```

## Groovy
### 环境
```bash
// java 需要切换到11 版本 https://stackoverflow.com/questions/52504825/how-to-install-jdk-11-under-ubuntu

wget https://groovy.jfrog.io/ui/native/dist-release-local/groovy-zips/apache-groovy-sdk-4.0.16.zip
unzip apache-groovy-sdk-4.0.16.zip
export PATH="/root/groovy-4.0.16/bin:$PATH"
```
### 执行
```
groovy test.groovy
```
## haskell
### 环境
```bash
curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | sh
ghcup install cabal
ghcup set cabal <version>  # 设置所安装 Cabal 的版本
cabal update
```
### 运行
```bash
ghc test.ts
./test
```


## Tcl

### 环境
```bash
apt-get install tcl
```

### 运行
```bash
tclsh test.tcl
```
## Scheme
### 环境
```bash
apt-get install racket
```

### 运行
```bash
racket test.scm
```
## Zig
### 环境
```bash
wget https://ziglang.org/builds/zig-linux-x86_64-0.12.0-dev.1849+bb0f7d55e.tar.xz
tar -xvf zig-linux-x86_64-0.12.0-dev.1849+bb0f7d55e.tar.xz
vi .bashrc
// 添加环境变量
source .bashrc
```
### 运行
```bash
zig test test.zig
```
## F\#
### 环境
```bash
apt-get install -y fsharp
```

### 运行
```bash
dotnet new console -lang "F#" -o MyFsharpApp
cd MyFsharpApp
dotnet build
dotnet run
```

## Racket
### 环境
```bash
apt-get install racket
```
### 运行
```bash
racket test.rkt
```

## Powershell
### 环境
```bash
apt-get install -y powershell
```
### 运行
```bash
pwsh test.ps1
```
## Tcsh
### 环境
```bash
apt-get install csh
```
### 运行
```bash
csh test.csh
```
## R
### 环境
```bash
apt-get install r-base
```
### 运行
```bash
Rscript test.R 
```


## Vb
### 环境
```bash
dotnet new console --language VB -o MyVbApp
dotnet nuget locals all --clear
```
### 运行
```bash
dotnet run
```
## elisp
### 环境
```bash
apt-get install emacs
```
### 运行
```bash
emacs --batch -l test.el
```
## erlang
### 环境
```bash
apt install erlang
```
### 运行
```bash
erlc has_close_elements.erl
erl -noshell -s has_close_elements test -s init stop
```
## Awk
## 安装
```
Linux系统自带
```
### 运行
```bash
awk -f test.awk data.txt
```


## Elixir
### 环境
```bash
apt-get install elixir
```
### 运行
```bash
 elixir test.exs
```


## Clisp
### 环境
```bash
apt-get install sbcl
```
### 运行
```bash
sbcl --script test.lisp
```


## Swift
### 环境
```bash
apt-get install clang libcurl4-openssl-dev libicu-dev libbsd-dev libblocksruntime-dev
（https://swift.org/download/）上获取最新版本的 Swift。
tar -xvzf swift-<version>-<platform>.tar.gz
export PATH=/path/to/swift-<version>-<platform>/usr/bin:"${PATH}"
swift --version
```
### 运行
```bash
swift 1.swift
```

