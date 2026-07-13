import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AlexLayout } from "@/alex/AlexLayout";
import Dashboard from "@/alex/Dashboard";
import Datasets from "@/alex/Datasets";
import Evaluation from "@/alex/Evaluation";
import Training from "@/alex/Training";
import TrainingJob from "@/alex/TrainingJob";

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AlexLayout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/setup" element={<Dashboard />} />
            <Route path="/datasets" element={<Datasets />} />
            <Route path="/training" element={<Training />} />
            <Route path="/training/:jobId" element={<TrainingJob />} />
            <Route path="/evaluation" element={<Evaluation />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </AlexLayout>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
