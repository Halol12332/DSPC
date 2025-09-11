1. Double click on the sln file (DSPC/SimpleMD/SimpleMolecularDynamics.sln)
2. Microsoft VS will be opened.
3. Ensure that you already tick the CUDA 12.9 in Build Dependencies -> Build Customization
4. Ctrl + ` to open terminal.
5. .\x64\Release-CPU\SimpleMolecularDynamics.exe 1024 200 --> to run the CPU test
6. .\x64\Release-GPU\SimpleMolecularDynamics.exe 1024 200 --> to run the GPU test

   Note : Above is just the benchmark..
