@echo off
setlocal

set "PY=python"
set "ENTRY=reader.py"
set "ICON=icon.ico"
set "DIST=dist"
set "BUILD=build"
set "ZIPNAME=SimpleComicReader_win64.zip"

set "NUITKA_CFLAGS=-O3 -march=native -fno-exceptions -fomit-frame-pointer"
set "NUITKA_LDFLAGS=-s"  rem strip symbols from final link (gcc option)
set "NUITKA_LTO=yes"

@REM set "INCLUDE_DIRS="
set "INCLUDE_FILES=icon-512.png"

mkdir "%DIST%"

%PY% -m nuitka ^
  --mingw64 ^
  --lto=yes ^
  --show-scons ^
  --standalone ^
  --enable-plugin=pyside6^
  --windows-console-mode=disable ^
  --windows-icon-from-ico=%ICON% ^
  --output-dir="%DIST%" ^
  %ENTRY%

if errorlevel 1 (
  echo Build failed.
  pause
  exit /b 1
)

set "DIST_DIR=%DIST%\%ENTRY%.dist"
if not exist "%DIST_DIR%" (
  for /f "delims=" %%D in ('dir /b /ad "%DIST%\*.dist" 2^>nul') do (
    set "DIST_DIR=%DIST%\%%D"
    goto :found_dist
  )
  echo Cannot find .dist directory.
  pause
  exit /b 1
)
:found_dist

if defined INCLUDE_FILES (
  for %%F in (%INCLUDE_FILES%) do if exist "%%~fF" copy /y "%%~fF" "%DIST_DIR%\"
)
if defined INCLUDE_DIRS (
  for %%D in (%INCLUDE_DIRS%) do if exist "%%~fD" xcopy "%%~fD" "%DIST_DIR%\%%~nD" /e /i /y >nul
)

if exist "%DIST_DIR%\cv2\opencv_videoio_ffmpeg4110_64.dll" (
  del /q "%DIST_DIR%\cv2\opencv_videoio_ffmpeg4110_64.dll"
  echo Removed opencv_videoio_ffmpeg4110_64.dll
)
if exist "%DIST_DIR%\qt6pdf.dll" (
  del /q "%DIST_DIR%\qt6pdf.dll"
  echo Removed qt6pdf.dll
)

if exist "%ICON%" copy /y "%ICON%" "%DIST_DIR%\" >nul

if exist "%DIST%\%ZIPNAME%" del /q "%DIST%\%ZIPNAME%"
powershell -Command "Compress-Archive -Path '%DIST_DIR%\\*' -DestinationPath '%DIST%\\%ZIPNAME%' -Force"

if errorlevel 1 (
  echo Zip failed; directory left at %DIST_DIR%
  echo Build completed but zipping failed.
) else (
  echo Build and zip successful: %DIST%\%ZIPNAME%
)

pause
endlocal
