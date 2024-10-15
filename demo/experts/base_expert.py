# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod


class BaseExpert(ABC):
    """Base class for all expert models."""
    @abstractmethod
    def mentioned_by(self, input: str):
        """Check if the expert is mentioned in the input."""
        raise NotImplementedError

    @abstractmethod
    def run(self, *args, **kwargs):
        """Run the expert model."""
        raise NotImplementedError
