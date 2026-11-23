{
  description = "vane — Lean 4 compiler and safety verifier for Linux accel/ane";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f system nixpkgs.legacyPackages.${system});
    in
    {
      packages = forAllSystems (system: pkgs:
        let
          vane = pkgs.stdenv.mkDerivation (finalAttrs: {
            pname = "vane";
            version = "0.1.0";
            src = self;

            nativeBuildInputs = [ pkgs.lean4 ];
            dontConfigure = true;

            buildPhase = ''
              runHook preBuild
              export HOME="$TMPDIR"
              lake build
              runHook postBuild
            '';

            installPhase = ''
              runHook preInstall
              install -Dm755 .lake/build/bin/vane "$out/bin/vane"
              install -Dm755 .lake/build/bin/vane-test "$out/bin/vane-test"
              runHook postInstall
            '';

            meta = with pkgs.lib; {
              description = "Lean 4 compiler and safety verifier for Linux accel/ane";
              mainProgram = "vane";
              platforms = systems;
            };
          });
        in
        {
          default = vane;
          inherit vane;
        });

      devShells = forAllSystems (system: pkgs:
        let
          pythonEnv = pkgs.python3.withPackages (ps: [ ps.numpy ]);
        in
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.lean4
              pythonEnv
            ];
          };
        });

      checks = forAllSystems (system: pkgs:
        let
          vanePkg = self.packages.${system}.default;
          pythonEnv = pkgs.python3.withPackages (ps: [ ps.numpy ]);
        in
        {
          lean-tests = pkgs.runCommand "vane-lean-tests" { } ''
            ${vanePkg}/bin/vane-test
            touch $out
          '';

          python-tests = pkgs.runCommand "vane-python-tests" {
            nativeBuildInputs = [ pythonEnv ];
          } ''
            cp -r ${self} work
            chmod -R u+w work
            cd work
            VANE_BIN=${vanePkg}/bin/vane python3 -m unittest discover -s tests -v
            touch $out
          '';
        });
    };
}
