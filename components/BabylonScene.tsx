import SceneComponent from "babylonjs-hook";
import {
  Vector3,
  Axis,
  Space,
  Color3,
  ExecuteCodeAction,
  Mesh,
  Vector4,
  Color4,
  ActionManager,
  FreeCamera,
  DirectionalLight,
  HemisphericLight,
  Scene,
  StandardMaterial,
  MeshBuilder,
  Texture,
  SolidParticleSystem,
} from "@babylonjs/core";
import { useAfterRender } from "react-babylonjs";
import * as CANNON from "cannon";
import { css } from "@emotion/react";
import { createCar } from "../utils/babylon/car";
import { ground } from "../utils/babylon/ground";
import "babylonjs-loaders";
window.CANNON = CANNON;

export default function BabylonScene() {
  const onSceneReady = (scene: Scene) => {
    // This creates and positions a free camera (non-mesh)
    var camera = new FreeCamera("camera1", new Vector3(0, 400, 0), scene);

    // This targets the camera to scene origin
    camera.setTarget(Vector3.Zero());

    // This attaches the camera to the canvas
    camera.attachControl(scene, true);
    // lights
    // new DirectionalLight("light1", new Vector3(0, 400, 0), scene);
    let light2 = new HemisphericLight("light2", new Vector3(0, 400, 0), scene);
    light2.intensity = 0.75;

    /*****************************Add Ground********************************************/
    ground(scene);
    /*****************************End Add Ground********************************************/

    const [
      pivotFI,
      pivotFO,
      pivot,
      carBody,
      wheelFI,
      wheelFO,
      wheelRI,
      wheelRO,
    ] = createCar(scene, {});

    /****************************Key Controls************************************************/

    let map = {}; //object for multiple key presses
    scene.actionManager = new ActionManager(scene);

    scene.actionManager.registerAction(
      new ExecuteCodeAction(ActionManager.OnKeyDownTrigger, function (evt) {
        map[evt.sourceEvent.key] = evt.sourceEvent.type == "keydown";
      })
    );

    scene.actionManager.registerAction(
      new ExecuteCodeAction(ActionManager.OnKeyUpTrigger, function (evt) {
        map[evt.sourceEvent.key] = evt.sourceEvent.type == "keydown";
      })
    );

    /****************************End Key Controls************************************************/

    /****************************Variables************************************************/

    let theta = 0;
    let deltaTheta = 0;
    let drive = 0; //distance translated per second
    let R = 50; //turning radius, initial set at pivot z value
    let NR: number; //Next turning radius on wheel turn
    let A = 4; // axel length
    let L = 4; //distance between wheel pivots
    let r = 1.5; // wheel radius
    let psi, psiRI, psiRO, psiFI, psiFO; //wheel rotations
    let phi; //rotation of car when turning

    let F: number; // frames per second

    /****************************End Variables************************************************/

    /****************************Animation******************************************************/

    scene.registerAfterRender(function () {
      F = scene.getEngine().getFps();
      if (drive > 0.15) {
        drive -= 0.15;
      } else if (drive < -0.15) {
        drive += 0.15;
      } else {
        drive = 0;
      }
      const distance = drive / F;
      psi = drive / (r * F);
      if ((map["a"] || map["A"]) && -Math.PI / 6 < theta) {
        deltaTheta = -Math.PI / 252;
        theta += deltaTheta;
        pivotFI.rotate(Axis.Y, deltaTheta, Space.LOCAL);
        pivotFO.rotate(Axis.Y, deltaTheta, Space.LOCAL);
        if (Math.abs(theta) > 0.00000001) {
          NR = A / 2 + L / Math.tan(theta);
        } else {
          theta = 0;
          NR = 0;
        }
        pivot.translate(Axis.Z, NR - R, Space.LOCAL);
        carBody.translate(Axis.Z, R - NR, Space.LOCAL);
        R = NR;
      }
      if ((map["d"] || map["D"]) && theta < Math.PI / 6) {
        deltaTheta = Math.PI / 252;
        theta += deltaTheta;
        pivotFI.rotate(Axis.Y, deltaTheta, Space.LOCAL);
        pivotFO.rotate(Axis.Y, deltaTheta, Space.LOCAL);
        if (Math.abs(theta) > 0.00000001) {
          NR = A / 2 + L / Math.tan(theta);
        } else {
          theta = 0;
          NR = 0;
        }
        pivot.translate(Axis.Z, NR - R, Space.LOCAL);
        carBody.translate(Axis.Z, R - NR, Space.LOCAL);
        R = NR;
      }
      if (map["w"] || map["W"]) {
        drive = 10;
      }
      if (map["s"] || map["S"]) {
        drive = -10;
      }
      if (Math.abs(drive) > 0) {
        phi = drive / (R * F);
        if (Math.abs(theta) > 0) {
          pivot.rotate(Axis.Y, phi, Space.WORLD);
          psiRI = drive / (r * F);
          psiRO = (drive * (R + A)) / (r * F);
          psiFI = (drive * Math.sqrt(R * R + L * L)) / (r * F);
          psiFO = (drive * Math.sqrt((R + A) * (R + A) + L * L)) / (r * F);
          wheelFI.rotate(Axis.Y, psiFI, Space.LOCAL);
          wheelFO.rotate(Axis.Y, psiFO, Space.LOCAL);
          wheelRI.rotate(Axis.Y, psiRI, Space.LOCAL);
          wheelRO.rotate(Axis.Y, psiRO, Space.LOCAL);
        } else {
          pivot.translate(Axis.X, -distance, Space.LOCAL);
          wheelFI.rotate(Axis.Y, psi, Space.LOCAL);
          wheelFO.rotate(Axis.Y, psi, Space.LOCAL);
          wheelRI.rotate(Axis.Y, psi, Space.LOCAL);
          wheelRO.rotate(Axis.Y, psi, Space.LOCAL);
        }
      }
    });
  };

  return (
    <SceneComponent
      antialias
      onSceneReady={onSceneReady}
      id="my-canvas"
      css={css`
        width: 100%;
        height: 100%;
        padding: 100px;
      `}
    />
  );
}
