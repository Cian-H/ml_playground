{
  inputs = {
    utils.url = "github:numtide/flake-utils";
  };

  inputs.nixpkgs.url = "github:NixOS/nixpkgs";

  outputs = { self, nixpkgs, utils }: utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs {
        system = "x86_64-linux";
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };
    in
    {
      devShell = pkgs.mkShell
        {
          NIX_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath ([
            pkgs.stdenv.cc.cc
            pkgs.zlib
            pkgs.libGL
            pkgs.libGLU
            pkgs.wlroots
            pkgs.ncurses5
            pkgs.linuxPackages.nvidia_x11
            pkgs.cudaPackages.cudnn
            pkgs.cudaPackages.cuda_nvml_dev
          ] ++ pkgs.cudatoolkit.all ++ pkgs.cudaPackages.cudatoolkit.all);
          NIX_LD = pkgs.lib.fileContents "${pkgs.stdenv.cc}/nix-support/dynamic-linker";

          buildInputs = with pkgs; [
            rye
            uv
          ];
        };
    }
  );
}

