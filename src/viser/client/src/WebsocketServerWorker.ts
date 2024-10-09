import { encode, decode } from "@msgpack/msgpack";
import { Message } from "./WebsocketMessages";
import AwaitLock from "await-lock";

export type WsWorkerIncoming =
  | { type: "send"; message: Message }
  | { type: "set_server"; server: string }
  | { type: "close" };

export type WsWorkerOutgoing =
  | { type: "connected" }
  | { type: "closed" }
  | { type: "message_batch"; messages: Message[] };

// Helper function to collect all ArrayBuffer objects. This is used for postMessage() move semantics.
function collectArrayBuffers(obj: any, buffers: Set<ArrayBuffer>) {
  if (obj instanceof ArrayBuffer) {
    buffers.add(obj);
  } else if (obj instanceof Uint8Array) {
    buffers.add(obj.buffer);
  } else if (obj && typeof obj === "object") {
    for (const key in obj) {
      if (Object.prototype.hasOwnProperty.call(obj, key)) {
        collectArrayBuffers(obj[key], buffers);
      }
    }
  }
  return buffers;
}
{
  let server: string | null = null;
  let ws: WebSocket | null = null;
  const orderLock = new AwaitLock();

  const postOutgoing = (
    data: WsWorkerOutgoing,
    transferable?: Transferable[],
  ) => {
    // @ts-ignore
    self.postMessage(data, transferable);
  };

  const tryConnect = () => {
    if (ws !== null) ws.close();
    ws = new WebSocket(server!);

    // Timeout is necessary when we're connecting to an SSH/tunneled port.
    const retryTimeout = setTimeout(() => {
      ws?.close();
    }, 5000);

    ws.onopen = () => {
      postOutgoing({ type: "connected" });
      clearTimeout(retryTimeout);
      console.log(`Connected! ${server}`);
    };

    ws.onclose = (event) => {
      postOutgoing({ type: "closed" });
      console.log(`Disconnected! ${server} code=${event.code}`);
      clearTimeout(retryTimeout);

      // Try to reconnect.
      if (server !== null) setTimeout(tryConnect, 1000);
    };

    ws.onmessage = async (event) => {
      // Reduce websocket backpressure.
      const messagePromise = new Promise<Message[]>((resolve) => {
        (event.data.arrayBuffer() as Promise<ArrayBuffer>).then((buffer) => {
          resolve(decode(new Uint8Array(buffer)) as Message[]);
        });
      });

      // Try our best to handle messages in order. If this takes more than 1 second, we give up. :)
      await orderLock.acquireAsync({ timeout: 1000 }).catch(() => {
        console.log("Order lock timed out.");
        orderLock.release();
      });
      try {
        const messages = await messagePromise;
        const arrayBuffers = collectArrayBuffers(messages, new Set());
        postOutgoing(
          { type: "message_batch", messages: messages },
          Array.from(arrayBuffers),
        );
      } finally {
        orderLock.acquired && orderLock.release();
      }
    };
  };

  self.onmessage = (e) => {
    const data: WsWorkerIncoming = e.data;

    if (data.type === "send") {
      ws!.send(encode(data.message));
    } else if (data.type === "set_server") {
      server = data.server;
      tryConnect();
    } else if (data.type == "close") {
      server = null;
      ws !== null && ws.close();
      self.close();
    } else {
      console.log(
        `WebSocket worker: got ${data}, not sure what to do with it!`,
      );
    }
  };
}
