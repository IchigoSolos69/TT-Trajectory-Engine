export interface DesktopContext {
  phase: 'idle'|'starting'|'ready'|'error';
  backendUrl: string|null;
  mode: 'live'|'simulation'|null;
  error: string|null;
  appVersion?: string;
}
export interface DesktopAPI {
  getContext(): Promise<DesktopContext>;
  openRecording(): Promise<DesktopContext>;
  startLive(): Promise<DesktopContext>;
  endSession(): Promise<void>;
  showDataDirectory(): Promise<void>;
  getReviewDrafts(): Promise<Record<string,unknown>>;
  saveReviewDrafts(value: Record<string,unknown>): Promise<void>;
  onSessionChanged(callback: (context: DesktopContext) => void): () => void;
}
declare global { interface Window { ttDesktop?: DesktopAPI; } }
