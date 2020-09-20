import React, { Component } from "react";
import * as tf from '@tensorflow/tfjs';
// import * as cocoSsd from "@tensorflow-models/coco-ssd";
import "./App.css";
// import {loadGraphModel} from '@tensorflow/tfjs-converter';

const MODEL_URL = '/model/model.json';


// const cat = document.getElementById('cat');
// model.execute(tf.browser.fromPixels(cat));

class App extends Component {
  state = {
    model: null,
    stream: null,
    videoElement: null,
    canvasElement: null
  };
  async componentDidMount() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user"
        },
        audio: false
      });

      // const model = await cocoSsd.load();
      const model = await tf.loadGraphModel(MODEL_URL, { strict: false });
      console.log(model);

      await this.setState({
        videoElement: this.refs.video,
        canvasElement: this.refs.canvas,
        stream,
        model
      });

      this.state.videoElement.srcObject = this.state.stream;
      this.predictFrame();
    } catch (err) {
      console.log(err);
    }
  }

  predictFrame = async () => {
    const canvas = document.getElementById('canvas');
    const image = tf.browser.fromPixels(canvas);
    const inputTensor = tf.cast(image, 'int32');
    const tensor4d = inputTensor.reshape([1, ...inputTensor.shape]);
    let predictions = await this.state.model.executeAsync(
      { 'image_tensor' : tf.zeros([1,300,300,3], 'int32') }, ['detection_boxes','detection_scores','detection_classes','num_detections']);
    tf.print(predictions);
    // this.drawPredictions(predictions);
    //recursive call
    // this.predictFrame();
  };

  drawPredictions = predictions => {
    const ctx = this.state.canvasElement.getContext("2d");
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    const font = "16px sans-serif";
    ctx.font = font;
    ctx.textBaseline = "top";
    predictions.forEach(prediction => {
      const x = prediction.bbox[0];
      const y = prediction.bbox[1];
      const width = prediction.bbox[2];
      const height = prediction.bbox[3];
      // Draw prediction box.
      ctx.strokeStyle = "#fa00ff";
      ctx.lineWidth = 4;
      ctx.strokeRect(x, y, width, height);
      // Draw text box.
      ctx.fillStyle = "#fa00ff";
      const textWidth = ctx.measureText(prediction.class).width;
      const textHeight = parseInt(font, 10); // base 10
      ctx.fillRect(x, y, textWidth + 4, textHeight + 4);
      // Draw text.
      ctx.fillStyle = "#000000";
      ctx.fillText(prediction.class, x, y);
    });
  };

  render() {
    return (
      <div>
        <video
          className="position"
          autoPlay
          playsInline
          muted
          ref="video"
          width="300"
          height="300"
        />
        <canvas id="canvas" className="position" ref="canvas" width="300" height="300" />
      </div>
    );
  }
}

export default App;
