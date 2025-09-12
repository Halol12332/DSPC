@echo off
setlocal
set OUT=results.csv
> "%OUT%" echo CSV,mode,N,steps,L,dt,rc,rs,time_ms,E_drift%%
for %%N in (500 1000 2000 5000 10000) do (
  >> "%OUT%" .\x64\Release-CPU\SimpleMolecularDynamics.exe %%N 2000
)
for %%N in (500 1000 2000 5000 10000) do (
  >> "%OUT%" .\x64\Release-GPU\SimpleMolecularDynamics.exe %%N 2000
)
echo Wrote %OUT%
